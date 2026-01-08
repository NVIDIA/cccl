# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Simple Numba-based iterator implementations.

This module contains straightforward iterator implementations that wrap
pointers or simple values. These include:
- RawPointer: Direct pointer wrapper
- CacheModifiedPointer: Pointer with cache-modified loads
- ConstantIterator: Always returns the same value
- CountingIterator: Returns sequential values
- DiscardIterator: Ignores writes
- ReverseIterator: Wraps another iterator in reverse
"""

import ctypes

import numba
import numpy as np
from llvmlite import ir
from numba import cuda, types
from numba.core.extending import intrinsic
from numba.core.typing.ctypes_utils import to_ctypes

from ..._utils.protocols import (
    compute_c_contiguous_strides_in_bytes,
    get_data_pointer,
    get_dtype,
    get_shape,
)
from ...typing import DeviceArrayLike
from .base import IteratorBase, IteratorKind

_DEVICE_POINTER_BITWIDTH = 64


# =============================================================================
# RawPointer
# =============================================================================


class RawPointerKind(IteratorKind):
    pass


class RawPointer(IteratorBase):
    iterator_kind_type = RawPointerKind

    def __init__(self, ptr: int, value_type: types.Type, obj: object):
        cvalue = ctypes.c_void_p(ptr)
        state_type = types.CPointer(value_type)
        self.obj = obj  # the container holding the data
        super().__init__(
            cvalue=cvalue,
            state_type=state_type,
            value_type=value_type,
        )

    @property
    def host_advance(self):
        return self._advance

    @property
    def advance(self):
        return self._advance

    @staticmethod
    def _advance(state, distance):
        state[0] = state[0] + distance

    @staticmethod
    def input_dereference(state, result):
        result[0] = state[0][0]

    @staticmethod
    def output_dereference(state, x):
        state[0][0] = x


def pointer(container, value_type: types.Type) -> RawPointer:
    return RawPointer(
        container.__cuda_array_interface__["data"][0],
        value_type,
        container,
    )


# =============================================================================
# CacheModifiedPointer
# =============================================================================


@intrinsic
def load_cs(typingctx, base):
    # Corresponding to `LOAD_CS` here:
    # https://nvidia.github.io/cccl/cub/api/classcub_1_1CacheModifiedInputIterator.html
    def codegen(context, builder, sig, args):
        rt = context.get_value_type(sig.return_type)
        if rt is None:
            raise RuntimeError(f"Unsupported return type: {type(sig.return_type)}")
        ftype = ir.FunctionType(rt, [rt.as_pointer()])
        bw = sig.return_type.bitwidth
        asm_txt = f"ld.global.cs.b{bw} $0, [$1];"
        if bw < 64:
            constraint = "=r, l"
        else:
            constraint = "=l, l"
        asm_ir = ir.InlineAsm(ftype, asm_txt, constraint)
        return builder.call(asm_ir, args)

    return base.dtype(base), codegen


class CacheModifiedPointerKind(IteratorKind):
    pass


class CacheModifiedPointer(IteratorBase):
    iterator_kind_type = CacheModifiedPointerKind

    def __init__(self, ptr: int, ntype: types.Type):
        cvalue = ctypes.c_void_p(ptr)
        value_type = ntype
        state_type = types.CPointer(value_type)
        super().__init__(
            cvalue=cvalue,
            state_type=state_type,
            value_type=value_type,
        )

    @property
    def host_advance(self):
        return self._advance

    @property
    def advance(self):
        return self._advance

    @property
    def input_dereference(self):
        return self._input_dereference

    @property
    def output_dereference(self):
        raise AttributeError(
            "CacheModifiedPointer cannot be used as an output iterator"
        )

    @staticmethod
    def _advance(state, distance):
        state[0] = state[0] + distance

    @staticmethod
    def _input_dereference(state, result):
        result[0] = load_cs(state[0])


# =============================================================================
# ConstantIterator
# =============================================================================


class ConstantIteratorKind(IteratorKind):
    pass


class ConstantIterator(IteratorBase):
    iterator_kind_type = ConstantIteratorKind

    def __init__(self, value: np.number):
        value_type = numba.from_dtype(value.dtype)
        cvalue = to_ctypes(value_type)(value)
        state_type = value_type
        super().__init__(
            cvalue=cvalue,
            state_type=state_type,
            value_type=value_type,
        )

    @property
    def host_advance(self):
        return self._advance

    @property
    def advance(self):
        return self._advance

    @property
    def input_dereference(self):
        return self._input_dereference

    @property
    def output_dereference(self):
        raise AttributeError("ConstantIterator cannot be used as an output iterator")

    @staticmethod
    def _advance(state, distance):
        pass

    @staticmethod
    def _input_dereference(state, result):
        result[0] = state[0]


# =============================================================================
# CountingIterator
# =============================================================================


class CountingIteratorKind(IteratorKind):
    pass


class CountingIterator(IteratorBase):
    iterator_kind_type = CountingIteratorKind

    def __init__(self, value: np.number):
        value_type = numba.from_dtype(value.dtype)
        cvalue = to_ctypes(value_type)(value)
        state_type = value_type
        super().__init__(
            cvalue=cvalue,
            state_type=state_type,
            value_type=value_type,
        )

    @property
    def host_advance(self):
        return self._advance

    @property
    def advance(self):
        return self._advance

    @property
    def input_dereference(self):
        return self._input_dereference

    @property
    def output_dereference(self):
        raise AttributeError("CountingIterator cannot be used as an output iterator")

    @staticmethod
    def _advance(state, distance):
        state[0] = state[0] + distance

    @staticmethod
    def _input_dereference(state, result):
        result[0] = state[0]


# =============================================================================
# DiscardIterator
# =============================================================================


class DiscardIteratorKind(IteratorKind):
    pass


class DiscardIterator(IteratorBase):
    iterator_kind_type = DiscardIteratorKind

    def __init__(self, reference_iterator=None):
        from ..._utils.temp_storage_buffer import TempStorageBuffer

        if reference_iterator is None:
            reference_iterator = TempStorageBuffer(1)

        if hasattr(reference_iterator, "__cuda_array_interface__"):
            iter = RawPointer(
                reference_iterator.__cuda_array_interface__["data"][0],
                numba.from_dtype(get_dtype(reference_iterator)),
                reference_iterator,
            )
        else:
            iter = reference_iterator

        super().__init__(
            cvalue=iter.cvalue,
            state_type=iter.state_type,
            value_type=iter.value_type,
        )

    @property
    def host_advance(self):
        return self._advance

    @property
    def advance(self):
        return self._advance

    @property
    def input_dereference(self):
        return self._dereference

    @property
    def output_dereference(self):
        return self._dereference

    @staticmethod
    def _advance(state, distance):
        pass

    @staticmethod
    def _dereference(state, result):
        pass


# =============================================================================
# ReverseIterator
# =============================================================================


class ReverseIteratorKind(IteratorKind):
    pass


def _get_last_element_ptr(device_array) -> int:
    shape = get_shape(device_array)
    dtype = get_dtype(device_array)

    strides_in_bytes = device_array.__cuda_array_interface__["strides"]
    if strides_in_bytes is None:
        strides_in_bytes = compute_c_contiguous_strides_in_bytes(shape, dtype.itemsize)

    offset_to_last_element = sum(
        (dim_size - 1) * stride for dim_size, stride in zip(shape, strides_in_bytes)
    )

    ptr = get_data_pointer(device_array)
    return ptr + offset_to_last_element


def make_reverse_iterator(it: DeviceArrayLike | IteratorBase):
    if not hasattr(it, "__cuda_array_interface__") and not isinstance(it, IteratorBase):
        raise NotImplementedError(
            f"Reverse iterator is not implemented for type {type(it)}"
        )

    if hasattr(it, "__cuda_array_interface__"):
        last_element_ptr = _get_last_element_ptr(it)
        it = RawPointer(last_element_ptr, numba.from_dtype(get_dtype(it)), it)

    it_advance = cuda.jit(it.advance, device=True)
    it_input_dereference = (
        cuda.jit(it.input_dereference, device=True)
        if hasattr(it, "input_dereference")
        else None
    )
    it_output_dereference = (
        cuda.jit(it.output_dereference, device=True)
        if hasattr(it, "output_dereference")
        else None
    )

    class ReverseIterator(IteratorBase):
        iterator_kind_type = ReverseIteratorKind

        def __init__(self, it):
            self._it = it
            super().__init__(
                cvalue=it.cvalue,
                state_type=it.state_type,
                value_type=it.value_type,
            )
            self._kind = self.__class__.iterator_kind_type(
                (it.kind, it.value_type), it.state_type
            )

        @property
        def host_advance(self):
            return self._advance

        @property
        def advance(self):
            return self._advance

        @property
        def input_dereference(self):
            if it_input_dereference is None:
                raise AttributeError("This iterator is not an input iterator")
            return ReverseIterator._input_dereference

        @property
        def output_dereference(self):
            if it_output_dereference is None:
                raise AttributeError("This iterator is not an output iterator")
            return ReverseIterator._output_dereference

        @staticmethod
        def _advance(state, distance):
            return it_advance(state, -distance)

        @staticmethod
        def _input_dereference(state, result):
            it_input_dereference(state, result)

        @staticmethod
        def _output_dereference(state, x):
            it_output_dereference(state, x)

    return ReverseIterator(it)
