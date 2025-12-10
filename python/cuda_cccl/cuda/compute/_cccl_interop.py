# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from __future__ import annotations

import enum
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


class _IteratorIO(enum.Enum):
    INPUT = 0
    OUTPUT = 1


def _iterator_to_cccl_iter(it: IteratorBase, io_kind: _IteratorIO) -> Iterator:
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

    advance_abi_name, advance_ltoir = it.get_advance_ltoir()
    match io_kind:
        case _IteratorIO.INPUT:
            deref_abi_name, deref_ltoir = it.get_input_dereference_ltoir()
        case _IteratorIO.OUTPUT:
            deref_abi_name, deref_ltoir = it.get_output_dereference_ltoir()
        case _:
            raise ValueError(f"Invalid io_kind: {io_kind}")

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


def _to_cccl_iter(
    it: IteratorBase | DeviceArrayLike | None, io_kind: _IteratorIO
) -> Iterator:
    if it is None:
        return _none_to_cccl_iter()
    if isinstance(it, IteratorBase):
        return _iterator_to_cccl_iter(it, io_kind)
    return _device_array_to_cccl_iter(it)


def to_cccl_input_iter(array_or_iterator) -> Iterator:
    return _to_cccl_iter(array_or_iterator, _IteratorIO.INPUT)


def to_cccl_output_iter(array_or_iterator) -> Iterator:
    return _to_cccl_iter(array_or_iterator, _IteratorIO.OUTPUT)


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


class _ArgMode(enum.Enum):
    """How a void* argument should be handled in wrapper codegen."""

    LOAD = "load"  # Cast to typed pointer, load value
    PTR = "ptr"  # Cast to typed pointer, pass pointer directly
    STORE = "store"  # Cast to typed pointer, store return value here
    STATE_ARRAYS = "states"  # Packed array of data pointers -> build array structs


class _ArgSpec:
    """Specification for a wrapper argument."""

    __slots__ = ("numba_type", "mode", "state_info")

    def __init__(self, numba_type, mode: _ArgMode, state_info=None):
        self.numba_type = numba_type
        self.mode = mode
        # For STATE_ARRAYS: list of {shape, itemsize, strides}
        self.state_info = state_info


def _codegen_void_ptr_wrapper(
    context, builder, args, arg_specs, func_device, inner_sig
):
    """Generate LLVM IR for a void* wrapper function.

    This is the codegen implementation shared by all void* wrappers.
    It processes each argument according to its _ArgSpec mode, calls
    the inner function, and stores the result if needed.

    Args:
        context: Numba codegen context
        builder: LLVM IR builder
        args: LLVM values for the void* arguments
        arg_specs: List of _ArgSpec describing each argument
        func_device: The device function to call
        inner_sig: Numba signature for the inner function

    Returns:
        LLVM dummy value (for void return)
    """

    input_vals = []
    ret_ptr = None

    for i, spec in enumerate(arg_specs):
        arg = args[i]

        if spec.mode == _ArgMode.LOAD:
            # Cast void* to typed pointer and load value
            llvm_type = context.get_value_type(spec.numba_type)
            typed_ptr = builder.bitcast(arg, llvm_type.as_pointer())
            val = builder.load(typed_ptr)
            input_vals.append(val)

        elif spec.mode == _ArgMode.PTR:
            # Cast void* to typed pointer, pass pointer directly
            llvm_type = context.get_value_type(spec.numba_type.dtype)
            typed_ptr = builder.bitcast(arg, llvm_type.as_pointer())
            input_vals.append(typed_ptr)

        elif spec.mode == _ArgMode.STORE:
            # Cast void* to typed pointer for storing result
            llvm_type = context.get_value_type(spec.numba_type)
            ret_ptr = builder.bitcast(arg, llvm_type.as_pointer())

    # Call the inner function
    cres = context.compile_subroutine(builder, func_device, inner_sig, caching=False)
    result = context.call_internal(builder, cres.fndesc, inner_sig, input_vals)

    # Store result if needed
    if ret_ptr is not None:
        builder.store(result, ret_ptr)

    return context.get_dummy_value()


def _create_void_ptr_wrapper_unified(func, name: str, arg_specs: list, inner_sig):
    """Unified wrapper creation for void* argument functions.

    Args:
        func: The function to wrap (will be compiled as device function)
        name: Base name for the wrapper function
        arg_specs: List of _ArgSpec describing each void* argument
        inner_sig: Numba signature for the inner function call

    Returns:
        Tuple of (wrapper_func, wrapper_sig)
    """
    from numba.cuda import jit as cuda_jit

    # Wrap function as device function
    func_device = cuda_jit(device=True)(func)

    # Generate argument names and signature
    arg_names = [f"arg_{i}" for i in range(len(arg_specs))]
    arg_str = ", ".join(arg_names)
    void_sig = types.void(*(types.voidptr for _ in arg_specs))

    # Create unique wrapper name
    unique_suffix = hex(id(func))[2:]
    wrapper_name = f"wrapped_{name}_{unique_suffix}"

    # We need exec() here because Numba's @intrinsic decorator requires:
    # 1. A function with a specific signature visible at parse time
    # 2. The number of arguments must match the wrapper signature
    # The actual codegen logic is in _codegen_void_ptr_wrapper - this just
    # creates the minimal intrinsic shell that delegates to it.
    wrapper_src = textwrap.dedent(f"""
    @intrinsic
    def impl(typingctx, {arg_str}):
        def codegen(context, builder, impl_sig, args):
            return codegen_helper(context, builder, args, arg_specs, func_device, inner_sig)
        return void_sig, codegen

    def {wrapper_name}({arg_str}):
        return impl({arg_str})
    """)

    local_dict = {
        "intrinsic": intrinsic,
        "void_sig": void_sig,
        "arg_specs": arg_specs,
        "func_device": func_device,
        "inner_sig": inner_sig,
        "codegen_helper": _codegen_void_ptr_wrapper,
    }
    exec(wrapper_src, globals(), local_dict)

    wrapper_func = local_dict[wrapper_name]
    wrapper_func.__globals__.update(local_dict)

    return wrapper_func, void_sig


def _create_op_void_ptr_wrapper(op, sig):
    """Creates a wrapper function that takes all void* arguments and calls the original operator.

    The wrapper takes N+1 arguments where N is the number of input arguments to `op`, the last
    argument is a pointer to the result.
    """
    arg_specs = [_ArgSpec(t, _ArgMode.LOAD) for t in sig.args]
    arg_specs.append(_ArgSpec(sig.return_type, _ArgMode.STORE))
    return _create_void_ptr_wrapper_unified(op, op.__name__, arg_specs, sig)


def _create_advance_void_ptr_wrapper(advance_fn, state_ptr_type):
    """Creates a wrapper function for iterator advance that takes void* arguments.

    The wrapper takes 2 void* arguments:
    - state pointer
    - offset pointer (points to uint64 value)
    """
    arg_specs = [
        _ArgSpec(state_ptr_type, _ArgMode.PTR),
        _ArgSpec(types.uint64, _ArgMode.LOAD),
    ]
    inner_sig = types.void(state_ptr_type, types.uint64)
    return _create_void_ptr_wrapper_unified(
        advance_fn, advance_fn.__name__, arg_specs, inner_sig
    )


def _create_input_dereference_void_ptr_wrapper(deref_fn, state_ptr_type, value_type):
    """Creates a wrapper function for input iterator dereference that takes void* arguments.

    The wrapper takes 2 void* arguments:
    - state pointer
    - result pointer (function writes result here)
    """
    arg_specs = [
        _ArgSpec(state_ptr_type, _ArgMode.PTR),
        _ArgSpec(types.CPointer(value_type), _ArgMode.PTR),
    ]
    inner_sig = types.void(state_ptr_type, types.CPointer(value_type))
    return _create_void_ptr_wrapper_unified(
        deref_fn, deref_fn.__name__, arg_specs, inner_sig
    )


def _create_output_dereference_void_ptr_wrapper(deref_fn, state_ptr_type, value_type):
    """Creates a wrapper function for output iterator dereference that takes void* arguments.

    The wrapper takes 2 void* arguments:
    - state pointer
    - value pointer (value to write)
    """
    arg_specs = [
        _ArgSpec(state_ptr_type, _ArgMode.PTR),
        _ArgSpec(value_type, _ArgMode.LOAD),
    ]
    inner_sig = types.void(state_ptr_type, value_type)
    return _create_void_ptr_wrapper_unified(
        deref_fn, deref_fn.__name__, arg_specs, inner_sig
    )


def to_stateless_cccl_op(op, sig: "Signature") -> Op:
    wrapped_op, wrapper_sig = _create_op_void_ptr_wrapper(op, sig)

    ltoir, _ = cuda.compile(wrapped_op, sig=wrapper_sig, output="ltoir")
    return Op(
        operator_type=OpKind.STATELESS,
        name=wrapped_op.__name__,
        ltoir=ltoir,
        state_alignment=1,
        state=None,
    )


def get_value_type(d_in: IteratorBase | DeviceArrayLike | GpuStruct | np.ndarray):
    from .struct import _Struct, gpu_struct

    if isinstance(d_in, IteratorBase):
        return d_in.value_type
    if isinstance(d_in, _Struct):
        return numba.typeof(d_in)
    dtype = get_dtype(d_in)
    if dtype.type == np.void:
        # we can't use the numba type corresponding to numpy struct
        # types directly, as those are passed by pointer to device
        # functions. Instead, we create an anonymous struct type
        # which has the appropriate pass-by-value semantics.
        return as_numba_type(gpu_struct(dtype))
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

    assert "LDL" not in sass, "LDL instruction found in SASS"
    assert "STL" not in sass, "STL instruction found in SASS"
    return temp_cubin_file.name


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
        temp_cubin_file_name = _check_compile_result(cubin)
        os.unlink(temp_cubin_file_name)

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
