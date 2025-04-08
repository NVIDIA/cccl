# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from __future__ import annotations

import functools
from typing import Callable, List

import numba
import numpy as np
from numba import cuda, types

from cuda.cccl import get_include_paths  # type: ignore[import-not-found]

from ._cy_bindings import (
    CommonData,
    IntEnum,
    Iterator,
    IteratorKind,
    IteratorState,
    IteratorStateView,
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

_TYPE_TO_ENUM = {
    types.int8: TypeEnum.INT8,
    types.int16: TypeEnum.INT16,
    types.int32: TypeEnum.INT32,
    types.int64: TypeEnum.INT64,
    types.uint8: TypeEnum.UINT8,
    types.uint16: TypeEnum.UINT16,
    types.uint32: TypeEnum.UINT32,
    types.uint64: TypeEnum.UINT64,
    types.float32: TypeEnum.FLOAT32,
    types.float64: TypeEnum.FLOAT64,
}


def _type_to_enum(numba_type: types.Type) -> IntEnum:
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
    numba_type = it.numba_type
    state_type = it.state_type
    size = context.get_value_type(state_type).get_abi_size(context.target_data)
    iterator_state = memoryview(it.state)
    if not iterator_state.nbytes == size:
        raise ValueError(
            f"Iterator state size, {iterator_state.nbytes} bytes, for iterator type {type(it)} "
            f"does not match size of numba type, {size} bytes"
        )
    alignment = context.get_value_type(numba_type).get_abi_alignment(
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
        state=IteratorStateView(it.state, size, it),
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


def to_cccl_op(op: Callable, sig) -> Op:
    ltoir, _ = cuda.compile(op, sig=sig, output="ltoir")
    return Op(
        operator_type=OpKind.STATELESS,
        name=op.__name__,
        ltoir=ltoir,
        state_alignment=1,
        state=None,
    )


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
def get_paths() -> List[str]:
    paths = [f"-I{path}" for path in get_include_paths().as_tuple() if path is not None]
    return paths


def call_build(build_impl_fn: Callable, *args, **kwargs):
    """Calls given build_impl_fn callable while providing compute capability and paths

    Returns result of the call.
    """
    cc_major, cc_minor = cuda.get_current_device().compute_capability
    cub_path, thrust_path, libcudacxx_path, cuda_include_path = get_paths()
    common_data = CommonData(
        cc_major, cc_minor, cub_path, thrust_path, libcudacxx_path, cuda_include_path
    )
    error = build_impl_fn(
        *args,
        common_data,
        **kwargs,
    )
    return error
