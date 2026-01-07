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
import warnings
from typing import TYPE_CHECKING, Callable, List

import numpy as np

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
from .typing import DeviceArrayLike, GpuStruct

if TYPE_CHECKING:
    from numba.core.typing import Signature

# Numpy dtype to TypeEnum mapping (no Numba dependency)
_NUMPY_DTYPE_TO_ENUM = {
    np.dtype("int8"): TypeEnum.INT8,
    np.dtype("int16"): TypeEnum.INT16,
    np.dtype("int32"): TypeEnum.INT32,
    np.dtype("int64"): TypeEnum.INT64,
    np.dtype("uint8"): TypeEnum.UINT8,
    np.dtype("uint16"): TypeEnum.UINT16,
    np.dtype("uint32"): TypeEnum.UINT32,
    np.dtype("uint64"): TypeEnum.UINT64,
    np.dtype("float16"): TypeEnum.FLOAT16,
    np.dtype("float32"): TypeEnum.FLOAT32,
    np.dtype("float64"): TypeEnum.FLOAT64,
}


@functools.lru_cache(maxsize=None)
def _numpy_dtype_to_info(dtype: np.dtype) -> TypeInfo:
    """Convert a numpy dtype to TypeInfo without Numba."""
    if dtype in _NUMPY_DTYPE_TO_ENUM:
        return TypeInfo(dtype.itemsize, dtype.alignment, _NUMPY_DTYPE_TO_ENUM[dtype])
    return TypeInfo(dtype.itemsize, dtype.alignment, TypeEnum.STORAGE)


def _is_numba_iterator(it) -> bool:
    """Check if an object is a Numba-based iterator."""
    # Import here to avoid circular imports and Numba dependency at module level
    from .iterators._iterators import IteratorBase

    return isinstance(it, IteratorBase)


def _is_compiled_iterator(it) -> bool:
    """Check if an object is a CompiledIterator."""
    from .iterators._compiled_iterator import CompiledIterator

    return isinstance(it, CompiledIterator)


class _IteratorIO(enum.Enum):
    INPUT = 0
    OUTPUT = 1


def _device_array_to_cccl_iter(array: DeviceArrayLike) -> Iterator:
    """Convert a device array to a CCCL Iterator."""
    if not is_contiguous(array):
        raise ValueError("Non-contiguous arrays are not supported.")
    dtype = get_dtype(array)

    # Handle structured dtypes by creating a proper gpu_struct
    if dtype.type == np.void:
        # This path requires Numba for gpu_struct
        from numba.core.extending import as_numba_type

        from ._numba.interop import numba_type_to_info
        from ._numba.struct import gpu_struct

        numba_type = as_numba_type(gpu_struct(dtype))
        info = numba_type_to_info(numba_type)
    else:
        info = _numpy_dtype_to_info(dtype)

    state_info = _numpy_dtype_to_info(np.dtype(np.intp))
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
    """Create a null CCCL Iterator."""
    info = _numpy_dtype_to_info(np.dtype(np.uint8))
    return Iterator(info.alignment, IteratorKind.POINTER, Op(), Op(), info, state=None)


def _compiled_iterator_to_cccl_iter(it, io_kind: _IteratorIO) -> Iterator:
    """Convert a CompiledIterator to a CCCL Iterator."""
    is_output = io_kind == _IteratorIO.OUTPUT
    return it.to_cccl_iter(is_output=is_output)


def _numba_iterator_to_cccl_iter(it, io_kind: _IteratorIO) -> Iterator:
    """Convert a Numba-based iterator to a CCCL Iterator."""
    from ._numba.interop import _IteratorIO as NumbaIteratorIO
    from ._numba.interop import numba_iterator_to_cccl_iter

    numba_io_kind = (
        NumbaIteratorIO.OUTPUT
        if io_kind == _IteratorIO.OUTPUT
        else NumbaIteratorIO.INPUT
    )
    return numba_iterator_to_cccl_iter(it, numba_io_kind)


def _to_cccl_iter(it, io_kind: _IteratorIO) -> Iterator:
    """Convert an iterator or array to a CCCL Iterator with dispatch logic."""
    if it is None:
        return _none_to_cccl_iter()
    if _is_compiled_iterator(it):
        return _compiled_iterator_to_cccl_iter(it, io_kind)
    if _is_numba_iterator(it):
        return _numba_iterator_to_cccl_iter(it, io_kind)
    return _device_array_to_cccl_iter(it)


def to_cccl_input_iter(array_or_iterator) -> Iterator:
    return _to_cccl_iter(array_or_iterator, _IteratorIO.INPUT)


def to_cccl_output_iter(array_or_iterator) -> Iterator:
    return _to_cccl_iter(array_or_iterator, _IteratorIO.OUTPUT)


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
        info = _numpy_dtype_to_info(array_or_struct.dtype)
        return Value(info, array_or_struct.data.cast("B"))
    else:
        # it's a GpuStruct, use the array underlying it
        return to_cccl_value(array_or_struct._data)


def to_stateless_cccl_op(op, sig: "Signature") -> Op:
    """Compile a Python callable to a CCCL Op using Numba."""
    from ._numba.interop import to_stateless_cccl_op as _to_stateless_cccl_op

    return _to_stateless_cccl_op(op, sig)


def get_value_type(d_in):
    """Get the value type for an input array, iterator, or struct.

    For CompiledIterator, returns the TypeDescriptor.
    For Numba-based iterators, returns a Numba type.
    For device arrays, returns a Numba type.
    """
    from .iterators._compiled_iterator import CompiledIterator

    # Handle CompiledIterator
    if isinstance(d_in, CompiledIterator):
        return d_in.value_type

    # Handle Numba-based iterators and arrays
    from ._numba.interop import get_value_type as _get_value_type

    return _get_value_type(d_in)


def set_cccl_iterator_state(cccl_it: Iterator, input_it):
    """Set the state of a CCCL Iterator from an input iterator or array."""
    if cccl_it.is_kind_pointer():
        ptr = get_data_pointer(input_it)
        ptr_obj = make_pointer_object(ptr, input_it)
        cccl_it.state = ptr_obj
    else:
        # Check if it's a CompiledIterator
        if _is_compiled_iterator(input_it):
            cccl_it.state = input_it.state
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

    from ._numba.interop import get_current_device_cc

    cc_major, cc_minor = get_current_device_cc()
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
    """Create a host-callable C function using Numba."""
    from ._numba.interop import make_host_cfunc

    return make_host_cfunc(state_ptr_ty, fn)


def cccl_iterator_set_host_advance(cccl_it: Iterator, array_or_iterator):
    """Set the host advance function for a CCCL Iterator."""
    if cccl_it.is_kind_iterator():
        # CompiledIterator doesn't support host_advance yet
        if _is_compiled_iterator(array_or_iterator):
            # CompiledIterator may not have host_advance
            if array_or_iterator._host_advance_fn is not None:
                # TODO: Support host advance for CompiledIterator
                raise NotImplementedError(
                    "host_advance for CompiledIterator is not yet supported"
                )
            return

        it = array_or_iterator
        fn_impl = it.host_advance
        if fn_impl is not None:
            cccl_it.host_advance_fn = _make_host_cfunc(it.state_ptr_type, fn_impl)
        else:
            raise ValueError(
                f"Iterator of type {type(it)} does not provide definition of host_advance function"
            )
