# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Numba-specific interop functions for cuda.compute.

This module contains all the Numba-dependent code that was previously
in _cccl_interop.py. It provides type conversion, iterator handling,
and op compilation using Numba.
"""

from __future__ import annotations

import enum
import functools
from typing import TYPE_CHECKING

import numba
from numba import cuda, types
from numba.core.extending import as_numba_type

from .._bindings import (
    Iterator,
    IteratorKind,
    Op,
    TypeEnum,
    TypeInfo,
)
from ..op import OpKind

if TYPE_CHECKING:
    from numba.core.typing import Signature

    from ..iterators._iterators import IteratorBase

# Mapping from Numba types to TypeEnum values
_NUMBA_TYPE_TO_ENUM = {
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


def _type_to_enum(numba_type: types.Type) -> TypeEnum:
    """Convert a Numba type to a TypeEnum value."""
    if numba_type in _NUMBA_TYPE_TO_ENUM:
        return _NUMBA_TYPE_TO_ENUM[numba_type]
    return TypeEnum.STORAGE


@functools.lru_cache(maxsize=None)
def numba_type_to_info(numba_type: types.Type) -> TypeInfo:
    """Convert a Numba type to a CCCL TypeInfo object."""
    context = cuda.descriptor.cuda_target.target_context
    value_type = context.get_value_type(numba_type)
    if isinstance(numba_type, types.Record):
        # then `value_type` is a pointer and we need the
        # alignment of the pointee.
        value_type = value_type.pointee
    size = value_type.get_abi_size(context.target_data)
    alignment = value_type.get_abi_alignment(context.target_data)
    return TypeInfo(size, alignment, _type_to_enum(numba_type))


class _IteratorIO(enum.Enum):
    """Direction for iterator dereference operations."""

    INPUT = 0
    OUTPUT = 1


def numba_iterator_to_cccl_iter(it: "IteratorBase", io_kind: _IteratorIO) -> Iterator:
    """Convert a Numba-based iterator to a CCCL Iterator object."""
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
        numba_type_to_info(it.value_type),
        state=it.state,
    )


def to_stateless_cccl_op(op, sig: "Signature") -> Op:
    """Compile a Python callable to a CCCL Op using Numba.

    Note: This function is now implemented in _numba/op.py. This is a
    re-export for backwards compatibility.
    """
    from .op import to_stateless_cccl_op as _impl

    return _impl(op, sig)


def make_host_cfunc(state_ptr_ty, fn):
    """Create a host-callable C function from a Python function using Numba."""
    sig = numba.void(state_ptr_ty, numba.int64)
    c_advance_fn = numba.cfunc(sig)(fn)
    return c_advance_fn.ctypes


def get_current_device_cc() -> tuple[int, int]:
    """Get the compute capability of the current CUDA device."""
    return cuda.get_current_device().compute_capability


def from_dtype(dtype):
    """Convert a numpy dtype to a Numba type."""
    return numba.from_dtype(dtype)


def typeof(val):
    """Get the Numba type of a Python value."""
    return numba.typeof(val)


def get_as_numba_type(cls):
    """Get the Numba type for a class registered with as_numba_type."""
    return as_numba_type(cls)


def get_value_type(d_in):
    """Get the value type for an input array, iterator, or struct.

    Args:
        d_in: Device array, iterator, or struct value

    Returns:
        Numba type representing the value type
    """
    import numpy as np

    from .._utils.protocols import get_dtype
    from ..struct import _Struct, gpu_struct
    from .iterators.base import IteratorBase

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
