# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numba
import functools
import ctypes
from numba import types, cuda


# MUST match `cccl_type_enum` in c/include/cccl/c/types.h
class TypeEnum(ctypes.c_int):
    INT8 = 0
    INT16 = 1
    INT32 = 2
    INT64 = 3
    UINT8 = 4
    UINT16 = 5
    UINT32 = 6
    UINT64 = 7
    FLOAT32 = 8
    FLOAT64 = 9
    STORAGE = 10


# MUST match `cccl_op_kind_t` in c/include/cccl/c/types.h
class OpKind(ctypes.c_int):
    STATELESS = 0
    STATEFUL = 1


# MUST match `cccl_iterator_kind_t` in c/include/cccl/c/types.h
class IteratorKind(ctypes.c_int):
    POINTER = 0
    ITERATOR = 1


# MUST match `cccl_type_info` in c/include/cccl/c/types.h
class TypeInfo(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_int),
        ("alignment", ctypes.c_int),
        ("type", TypeEnum),
    ]


# MUST match `cccl_op_t` in c/include/cccl/c/types.h
class Op(ctypes.Structure):
    _fields_ = [
        ("type", OpKind),
        ("name", ctypes.c_char_p),
        ("ltoir", ctypes.c_char_p),
        ("ltoir_size", ctypes.c_int),
        ("size", ctypes.c_int),
        ("alignment", ctypes.c_int),
        ("state", ctypes.c_void_p),
    ]


# MUST match `cccl_iterator_t` in c/include/cccl/c/types.h
class Iterator(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_int),
        ("alignment", ctypes.c_int),
        ("type", IteratorKind),
        ("advance", Op),
        ("dereference", Op),
        ("value_type", TypeInfo),
        ("state", ctypes.c_void_p),
    ]


# MUST match `cccl_device_reduce_build_result_t` in c/include/cccl/c/reduce.h
class DeviceReduceBuildResult(ctypes.Structure):
    _fields_ = [
        ("cc", ctypes.c_int),
        ("cubin", ctypes.c_void_p),
        ("cubin_size", ctypes.c_size_t),
        ("library", ctypes.c_void_p),
        ("accumulator_size", ctypes.c_ulonglong),
        ("single_tile_kernel", ctypes.c_void_p),
        ("single_tile_second_kernel", ctypes.c_void_p),
        ("reduction_kernel", ctypes.c_void_p),
    ]


# MUST match `cccl_value_t` in c/include/cccl/c/types.h
class Value(ctypes.Structure):
    _fields_ = [("type", TypeInfo), ("state", ctypes.c_void_p)]


def _type_to_enum(numba_type):
    mapping = {
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
    if numba_type in mapping:
        return mapping[numba_type]
    return TypeEnum.STORAGE


# TODO: replace with functools.cache once our docs build environment
# is upgraded to at least Python 3.9
@functools.lru_cache(maxsize=None)
def _numba_type_to_info(numba_type):
    context = cuda.descriptor.cuda_target.target_context
    value_type = context.get_value_type(numba_type)
    size = value_type.get_abi_size(context.target_data)
    alignment = value_type.get_abi_alignment(context.target_data)
    return TypeInfo(size, alignment, _type_to_enum(numba_type))


@functools.lru_cache(maxsize=None)
def _numpy_type_to_info(numpy_type):
    numba_type = numba.from_dtype(numpy_type)
    return _numba_type_to_info(numba_type)


def _device_array_to_cccl_iter(array):
    dtype = array.dtype
    info = _numpy_type_to_info(dtype)
    # Note: this is slightly slower, but supports all ndarray-like objects as long as they support CAI
    # TODO: switch to use gpumemoryview once it's ready
    return Iterator(
        info.size,
        info.alignment,
        IteratorKind.POINTER,
        Op(),
        Op(),
        info,
        array.__cuda_array_interface__["data"][0],
    )


def _iterator_to_cccl_iter(it):
    context = cuda.descriptor.cuda_target.target_context
    numba_type = it.numba_type
    size = context.get_value_type(numba_type).get_abi_size(context.target_data)
    alignment = context.get_value_type(numba_type).get_abi_alignment(
        context.target_data
    )
    (advance_abi_name, advance_ltoir), (deref_abi_name, deref_ltoir) = it.ltoirs.items()
    advance_op = Op(
        OpKind.STATELESS,
        advance_abi_name.encode("utf-8"),
        ctypes.c_char_p(advance_ltoir),
        len(advance_ltoir),
        1,
        1,
        None,
    )
    deref_op = Op(
        OpKind.STATELESS,
        deref_abi_name.encode("utf-8"),
        ctypes.c_char_p(deref_ltoir),
        len(deref_ltoir),
        1,
        1,
        None,
    )
    return Iterator(
        size,
        alignment,
        OpKind.STATEFUL,
        advance_op,
        deref_op,
        _numba_type_to_info(it.value_type),
        it.state,
    )


def type_enum_as_name(enum_value):
    assert isinstance(enum_value, int)
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


def to_cccl_iter(array_or_iterator):
    from cuda.parallel.experimental.iterators._iterators import IteratorBase

    if isinstance(array_or_iterator, IteratorBase):
        return _iterator_to_cccl_iter(array_or_iterator)
    return _device_array_to_cccl_iter(array_or_iterator)


def host_array_to_value(array):
    dtype = array.dtype
    info = _numpy_type_to_info(dtype)
    return Value(info, array.ctypes.data)
