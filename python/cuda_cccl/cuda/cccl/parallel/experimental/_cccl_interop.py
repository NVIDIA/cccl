# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from __future__ import annotations

import functools
import os
import subprocess
import tempfile
import textwrap
import warnings
from typing import TYPE_CHECKING, Callable, List

import numba
import numpy as np
from numba import cuda, types
from numba.core.extending import as_numba_type, intrinsic

# TODO: adding a type-ignore here because `cuda` being a
# namespace package confuses mypy when `cuda.<something_else>`
# is installed, but not `cuda.cccl`. For namespace packages,
# it appears we need to actually install the sub-package
# in order for mypy to find its py.typed file. However, CI
# does type checking of `cuda.cccl` without actually installing
# it.
#
# We need to find a better solution for this.
from cuda.cccl import get_include_paths  # type: ignore

from ._bindings import (
    CommonData,
    Iterator,
    IteratorKind,
    IteratorState,
    Op,
    Pointer,
    TypeEnum,
    TypeInfo,
    Value,
    make_pointer_object,
)
from ._utils.protocols import get_data_pointer, get_dtype, is_contiguous
from .iterators._iterators import IteratorBase
from .op import OpKind
from .typing import DeviceArrayLike, GpuStruct

_TYPE_TO_ENUM = {
    types.int8: TypeEnum.INT8,
    types.int16: TypeEnum.INT16,
    types.int32: TypeEnum.INT32,
    types.int64: TypeEnum.INT64,
    types.uint8: TypeEnum.UINT8,
    types.uint16: TypeEnum.UINT16,
    types.uint32: TypeEnum.UINT32,
    types.uint64: TypeEnum.UINT64,
    types.float16: TypeEnum.FLOAT16,
    types.float32: TypeEnum.FLOAT32,
    types.float64: TypeEnum.FLOAT64,
}


if TYPE_CHECKING:
    from numba.core.typing import Signature


def _type_to_enum(numba_type: types.Type) -> TypeEnum:
    if numba_type in _TYPE_TO_ENUM:
        return _TYPE_TO_ENUM[numba_type]
    return TypeEnum.STORAGE


# TODO: replace with functools.cache once our docs build environment
# is upgraded to at least Python 3.9
@functools.lru_cache(maxsize=None)
def _numba_type_to_info(numba_type: types.Type) -> TypeInfo:
    context = cuda.descriptor.cuda_target.target_context
    value_type = context.get_value_type(numba_type)
    if isinstance(numba_type, types.Record):
        # then `value_type` is a pointer and we need the
        # alignment of the pointee.
        value_type = value_type.pointee
    size = value_type.get_abi_size(context.target_data)
    alignment = value_type.get_abi_alignment(context.target_data)
    return TypeInfo(size, alignment, _type_to_enum(numba_type))


@functools.lru_cache(maxsize=None)
def _numpy_type_to_info(numpy_type: np.dtype) -> TypeInfo:
    numba_type = numba.from_dtype(numpy_type)
    return _numba_type_to_info(numba_type)


def _device_array_to_cccl_iter(array: DeviceArrayLike) -> Iterator:
    if not is_contiguous(array):
        raise ValueError("Non-contiguous arrays are not supported.")
    info = _numpy_type_to_info(get_dtype(array))
    state_info = _numpy_type_to_info(np.intp)
    return Iterator(
        state_info.alignment,
        IteratorKind.POINTER,
        Op(),
        Op(),
        info,
        # Note: this is slightly slower, but supports all ndarray-like objects
        # as long as they support CAI
        # TODO: switch to use gpumemoryview once it's ready
        state=get_data_pointer(array),
    )


def _iterator_to_cccl_iter(it: IteratorBase) -> Iterator:
    context = cuda.descriptor.cuda_target.target_context
    state_ptr_type = it.state_ptr_type
    state_type = it.state_type
    size = context.get_value_type(state_type).get_abi_size(context.target_data)
    iterator_state = memoryview(it.state)
    if not iterator_state.nbytes == size:
        raise ValueError(
            f"Iterator state size, {iterator_state.nbytes} bytes, for iterator type {type(it)} "
            f"does not match size of numba type, {size} bytes"
        )
    alignment = context.get_value_type(state_ptr_type).get_abi_alignment(
        context.target_data
    )
    (advance_abi_name, advance_ltoir), (deref_abi_name, deref_ltoir) = it.ltoirs.items()
    advance_op = Op(
        operator_type=OpKind.STATELESS,
        name=advance_abi_name,
        ltoir=advance_ltoir,
    )
    deref_op = Op(
        operator_type=OpKind.STATELESS,
        name=deref_abi_name,
        ltoir=deref_ltoir,
    )
    return Iterator(
        alignment,
        IteratorKind.ITERATOR,
        advance_op,
        deref_op,
        _numba_type_to_info(it.value_type),
        state=it.state,
    )


def _none_to_cccl_iter() -> Iterator:
    # Any type could be used here, we just need to pass NULL.
    info = _numpy_type_to_info(np.uint8)
    return Iterator(info.alignment, IteratorKind.POINTER, Op(), Op(), info, state=None)


def type_enum_as_name(enum_value: int) -> str:
    return (
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
        "STORAGE",
    )[enum_value]


def to_cccl_iter(array_or_iterator) -> Iterator:
    if array_or_iterator is None:
        return _none_to_cccl_iter()
    if isinstance(array_or_iterator, IteratorBase):
        return _iterator_to_cccl_iter(array_or_iterator)
    return _device_array_to_cccl_iter(array_or_iterator)


def to_cccl_value_state(array_or_struct: np.ndarray | GpuStruct) -> memoryview:
    if isinstance(array_or_struct, np.ndarray):
        assert array_or_struct.flags.contiguous
        data = array_or_struct.data.cast("B")
        return data
    else:
        # it's a GpuStruct, use the array underlying it
        return to_cccl_value_state(array_or_struct._data)


def to_cccl_value(array_or_struct: np.ndarray | GpuStruct) -> Value:
    if isinstance(array_or_struct, np.ndarray):
        info = _numpy_type_to_info(array_or_struct.dtype)
        return Value(info, array_or_struct.data.cast("B"))
    else:
        # it's a GpuStruct, use the array underlying it
        return to_cccl_value(array_or_struct._data)


def _create_void_ptr_wrapper(op, sig):
    """Creates a wrapper function that takes all void* arguments and calls the original operator.

    The wrapper takes N+1 arguments where N is the number of input arguments to `op`, the last
    argument is a pointer to the result.
    """
    # Generate argument names for both inputs and output
    input_args = [f"arg_{i}" for i in range(len(sig.args))]
    all_args = input_args + ["ret"]  # ret is the output pointer
    arg_str = ", ".join(all_args)
    void_sig = types.void(*(types.voidptr for _ in all_args))

    # Create the wrapper function source code
    wrapper_src = textwrap.dedent(f"""
    @intrinsic
    def impl(typingctx, {arg_str}):
        def codegen(context, builder, impl_sig, args):
            # Get LLVM types for all arguments
            arg_types = [context.get_value_type(t) for t in sig.args]
            ret_type = context.get_value_type(sig.return_type)

            # Bitcast from void* to the appropriate pointer types
            input_ptrs = [builder.bitcast(p, t.as_pointer()) for p, t in zip(args[:-1], arg_types)]
            ret_ptr = builder.bitcast(args[-1], ret_type.as_pointer())

            # Load input values from pointers
            input_vals = [builder.load(p) for p in input_ptrs]

            # Call the original operator
            result = context.compile_internal(builder, op, sig, input_vals)

            # Store the result
            builder.store(result, ret_ptr)

            return context.get_dummy_value()
        return void_sig, codegen

    # intrinsics cannot directly be compiled by numba, so we make a trivial wrapper:
    def wrapped_{op.__name__}({arg_str}):
        return impl({arg_str})
    """)

    # Create namespace and compile the wrapper
    local_dict = {
        "types": types,
        "sig": sig,
        "op": op,
        "intrinsic": intrinsic,
        "void_sig": void_sig,
    }
    exec(wrapper_src, globals(), local_dict)

    wrapper_func = local_dict[f"wrapped_{op.__name__}"]
    wrapper_func.__globals__.update(local_dict)

    return wrapper_func, void_sig


def to_cccl_op(op: Callable | OpKind, sig: Signature | None) -> Op:
    """Return an `Op` object corresponding to the given callable or well-known operation.

    For well-known operations (Ops), returns an Op with the appropriate
    kind and empty ltoir/state.

    For callables, wraps the callable in a device function that takes void* arguments
    and a void* return value. This is the only way to match the corresponding "extern"
    declaration of the device function in the C code, which knows nothing about the types
    of the arguments and return value. The two functions must have the same signature in
    order to link correctly without violating ODR.
    """
    # Check if op is a well-known operation
    if isinstance(op, OpKind):
        return Op(
            operator_type=op,
            name="",
            ltoir=b"",
            state_alignment=1,
            state=b"",
        )

    # op is a callable:
    wrapped_op, wrapper_sig = _create_void_ptr_wrapper(op, sig)

    ltoir, _ = cuda.compile(wrapped_op, sig=wrapper_sig, output="ltoir")
    return Op(
        operator_type=OpKind.STATELESS,
        name=wrapped_op.__name__,
        ltoir=ltoir,
        state_alignment=1,
        state=None,
    )


def get_value_type(d_in: IteratorBase | DeviceArrayLike):
    from .struct import gpu_struct_from_numpy_dtype

    if isinstance(d_in, IteratorBase):
        return d_in.value_type
    dtype = get_dtype(d_in)
    if dtype.type == np.void:
        # we can't use the numba type corresponding to numpy struct
        # types directly, as those are passed by pointer to device
        # functions. Instead, we create an anonymous struct type
        # which has the appropriate pass-by-value semantics.
        return as_numba_type(gpu_struct_from_numpy_dtype("anonymous", dtype))
    return numba.from_dtype(dtype)


def set_cccl_iterator_state(cccl_it: Iterator, input_it):
    if cccl_it.is_kind_pointer():
        ptr = get_data_pointer(input_it)
        ptr_obj = make_pointer_object(ptr, input_it)
        cccl_it.state = ptr_obj
    else:
        state_ = input_it.state
        if isinstance(state_, (IteratorState, Pointer)):
            cccl_it.state = state_
        else:
            cccl_it.state = make_pointer_object(state_, input_it)


@functools.lru_cache()
def get_includes() -> List[str]:
    def as_option(p):
        if p is None:
            return ""
        return f"-I{p}"

    paths = get_include_paths().as_tuple()
    opts = [as_option(path) for path in paths]
    return opts


def _check_compile_result(cubin: bytes):
    # check compiled code for LDL/STL instructions
    temp_cubin_file = tempfile.NamedTemporaryFile(delete=False)
    try:
        temp_cubin_file.write(cubin)
        out = subprocess.run(
            ["nvdisasm", "-gi", temp_cubin_file.name], capture_output=True
        )
        if out.returncode != 0:
            raise RuntimeError("nvdisasm failed")
        sass = out.stdout.decode("utf-8")
    except FileNotFoundError:
        sass = "nvdiasm not found, skipping SASS validation"
        warnings.warn(sass)
    finally:
        os.unlink(temp_cubin_file.name)

    assert "LDL" not in sass, "LDL instruction found in SASS"
    assert "STL" not in sass, "STL instruction found in SASS"


# this global variable controls whether the compile result is checked
# for LDL/STL instructions. Should be set to `True` for testing only.
_check_sass: bool = False


def call_build(build_impl_fn: Callable, *args, **kwargs):
    """Calls given build_impl_fn callable while providing compute capability and paths

    Returns result of the call.
    """
    global _check_sass

    cc_major, cc_minor = cuda.get_current_device().compute_capability
    cub_path, thrust_path, libcudacxx_path, cuda_include_path = get_includes()
    common_data = CommonData(
        cc_major, cc_minor, cub_path, thrust_path, libcudacxx_path, cuda_include_path
    )
    result = build_impl_fn(
        *args,
        common_data,
        **kwargs,
    )

    if _check_sass:
        cubin = result._get_cubin()
        _check_compile_result(cubin)

    return result


def _make_host_cfunc(state_ptr_ty, fn):
    sig = numba.void(state_ptr_ty, numba.int64)
    c_advance_fn = numba.cfunc(sig)(fn)

    return c_advance_fn.ctypes


def cccl_iterator_set_host_advance(cccl_it: Iterator, array_or_iterator):
    if cccl_it.is_kind_iterator():
        it = array_or_iterator
        fn_impl = it.host_advance
        if fn_impl is not None:
            cccl_it.host_advance_fn = _make_host_cfunc(it.state_ptr_type, fn_impl)
        else:
            raise ValueError(
                f"Iterator of type {type(it)} does not provide definition of host_advance function"
            )
