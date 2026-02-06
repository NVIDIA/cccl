# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
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
from typing import Callable, List

try:
    from cuda.core import Device as CudaDevice
except ImportError:
    from cuda.core.experimental import Device as CudaDevice


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

from . import types
from ._bindings import (
    CommonData,
    Iterator,
    IteratorKind,
    IteratorState,
    Op,
    OpKind,
    Pointer,
    TypeEnum,
    TypeInfo,
    Value,
    make_pointer_object,
)
from ._utils.protocols import get_data_pointer, get_dtype, is_contiguous
from .iterators._iterators import IteratorBase
from .typing import DeviceArrayLike, GpuStruct

# Mapping from numpy dtype to TypeEnum for creating TypeInfo
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
    np.dtype("bool"): TypeEnum.BOOLEAN,
}


@functools.lru_cache(maxsize=256)
def _type_info_from_dtype(dtype: np.dtype) -> TypeInfo:
    """
    Create a TypeInfo from a numpy dtype.
    Handles both primitive types and structured dtypes.
    """
    dtype = np.dtype(dtype)

    # Handle structured dtypes
    if dtype.type == np.void and dtype.fields is not None:
        return TypeInfo(dtype.itemsize, dtype.alignment, TypeEnum.STORAGE)

    if dtype.kind == "c":
        return TypeInfo(dtype.itemsize, dtype.alignment, TypeEnum.STORAGE)

    # Fallback for any other type
    type_enum = _NUMPY_DTYPE_TO_ENUM.get(dtype, TypeEnum.STORAGE)
    return TypeInfo(dtype.itemsize, dtype.alignment, type_enum)


def _is_well_known_op(op: OpKind) -> bool:
    return isinstance(op, OpKind) and op not in (OpKind.STATELESS, OpKind.STATEFUL)


def _device_array_to_cccl_iter(array: DeviceArrayLike) -> Iterator:
    if not is_contiguous(array):
        raise ValueError("Non-contiguous arrays are not supported.")
    dtype = get_dtype(array)

    info = _type_info_from_dtype(dtype)
    state_info = _type_info_from_dtype(np.intp)
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
    info = _type_info_from_dtype(np.uint8)
    return Iterator(info.alignment, IteratorKind.POINTER, Op(), Op(), info, state=None)


class _IteratorIO(enum.Enum):
    INPUT = 0
    OUTPUT = 1


def _to_cccl_iter(
    it: IteratorBase | DeviceArrayLike | None, io_kind: _IteratorIO
) -> Iterator:
    from ._jit import compile_iterator

    if it is None:
        return _none_to_cccl_iter()
    if isinstance(it, IteratorBase):
        io_kind_str = "input" if io_kind == _IteratorIO.INPUT else "output"
        return compile_iterator(it, io_kind_str)
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
        info = _type_info_from_dtype(array_or_struct.dtype)
        return Value(info, array_or_struct.data.cast("B"))
    else:
        # it's a GpuStruct, use the array underlying it
        return to_cccl_value(array_or_struct._data)


def set_cccl_value_state(cccl_value: Value, array_or_struct: np.ndarray | GpuStruct):
    """
    Set the state of a CCCL Value object from a numpy array or GpuStruct.

    Args:
        cccl_value: The CCCL Value binding object
        array_or_struct: The numpy array or GpuStruct to get the state from
    """
    cccl_value.state = to_cccl_value_state(array_or_struct)


def get_value_type(d_in: IteratorBase | DeviceArrayLike | GpuStruct | np.ndarray):
    from .struct import _Struct

    if isinstance(d_in, IteratorBase):
        return d_in.value_type

    if isinstance(d_in, _Struct):
        return type(d_in)._type_descriptor  # type: ignore[union-attr]

    dtype = get_dtype(d_in)

    if dtype.type == np.void:
        return types.from_numpy_dtype(dtype)

    return types.from_numpy_dtype(dtype)


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

    cc_major, cc_minor = CudaDevice().compute_capability
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
