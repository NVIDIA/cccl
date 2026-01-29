# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ctypes
from typing import Callable

import numpy as np
from numba import cuda

from .. import types
from .._bindings import IteratorState
from .._caching import CachableFunction, cache_with_registered_key_functions
from .._intrinsics import load_cs
from .._utils.protocols import (
    compute_c_contiguous_strides_in_bytes,
    get_data_pointer,
    get_dtype,
    get_shape,
)
from ..typing import DeviceArrayLike


class IteratorKind:
    # The `.kind` of an iterator encapsulates additional metadata about the iterator,
    # analogous to the `.dtype` of a NumPy array.
    def __init__(
        self, value_type: types.TypeDescriptor, state_type: types.TypeDescriptor
    ):
        self.value_type = value_type
        self.state_type = state_type

    def __repr__(self):
        return (
            f"{self.__class__.__name__}[{str(self.value_type), str(self.state_type)}]"
        )

    def __eq__(self, other):
        return (
            type(self) is type(other)
            and self.value_type == other.value_type
            and self.state_type == other.state_type
        )

    def __hash__(self):
        return hash((type(self), self.value_type, self.state_type))


class IteratorBase:
    """
    An Iterator is a wrapper around a state pointer, and must define the following:

    - an `advance` property returning a (static) method which receives the
      state pointer and performs an action that advances the state pointer by the offset
      `distance` (returns nothing).
    - `input_dereference` and `output_dereference` properties that return
      (static) methods for reading from and writing to the state pointer respectively.

    Iterators are not meant to be used directly. They are constructed and passed
    to algorithms which internally invoke their methods.

    The `advance`, `input_dereference`, and `output_dereference` must be compilable
    to device code by numba.
    """

    def __init__(
        self,
        kind: IteratorKind,
        cvalue,
        state_type: types.TypeDescriptor,
        value_type: types.TypeDescriptor,
    ):
        """
        Parameters
        ----------
        cvalue
          A ctypes type representing the object pointed to by the iterator.
        state_type
          A TypeDescriptor representing the state type
        value_type
          A TypeDescriptor representing the value type
        """
        self._kind = kind
        self.cvalue = cvalue
        self.state_type = state_type
        self.value_type = value_type
        self._state = IteratorState(self.cvalue)

    @property
    def kind(self):
        return self._kind

    @property
    def state(self) -> IteratorState:
        return self._state

    @property
    def advance(state):
        raise NotImplementedError("Subclasses must override advance property")

    @property
    def input_dereference(state):
        return None

    @property
    def output_dereference(state):
        return None

    @property
    def host_advance(self):
        return None

    @property
    def children(self):
        return ()

    @property
    def is_input_iterator(self) -> bool:
        return self.input_dereference is not None

    @property
    def is_output_iterator(self) -> bool:
        return self.output_dereference is not None

    def __add__(self, offset: int):
        # add the offset to the iterator's state, and return a new iterator
        # with the new state.
        res = type(self).__new__(type(self))
        res.state_type = self.state_type
        res.value_type = self.value_type
        res._kind = self._kind
        res.cvalue = type(self.cvalue)(self.cvalue.value + offset)
        res._state = IteratorState(res.cvalue)

        return res


class RawPointerKind(IteratorKind):
    pass


class RawPointer(IteratorBase):
    def __init__(self, ptr: int, value_type: types.TypeDescriptor, obj: object):
        """
        Args:
            ptr: Device pointer address
            value_type: A TypeDescriptor for the element type
            obj: Reference to the underlying object (to prevent garbage collection)
        """

        cvalue = ctypes.c_void_p(ptr)
        # state_type is a pointer to value_type
        state_type = value_type.pointer()
        self.obj = obj
        kind = RawPointerKind(value_type, state_type)

        super().__init__(
            kind=kind,
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


def pointer(container) -> RawPointer:
    """Create a RawPointer iterator from a device array."""

    return RawPointer(
        container.__cuda_array_interface__["data"][0],
        types.from_numpy_dtype(get_dtype(container)),
        container,
    )


class CacheModifiedPointerKind(IteratorKind):
    pass


class CacheModifiedPointer(IteratorBase):
    def __init__(self, ptr: int, value_type: types.TypeDescriptor):
        """
        Args:
            ptr: Device pointer address
            value_type: A TypeDescriptor for the element type
        """
        cvalue = ctypes.c_void_p(ptr)
        state_type = value_type.pointer()
        kind = CacheModifiedPointerKind(value_type, state_type)
        super().__init__(
            kind=kind,
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


class ConstantIteratorKind(IteratorKind):
    pass


class ConstantIterator(IteratorBase):
    def __init__(self, value: np.number):
        value_type = types.from_numpy_dtype(value.dtype)
        cvalue = types.to_ctypes_type(value_type)(value)
        # state_type is the value itself (not a pointer)
        state_type = value_type
        kind = ConstantIteratorKind(value_type, state_type)
        super().__init__(
            kind=kind,
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


class CountingIteratorKind(IteratorKind):
    pass


class CountingIterator(IteratorBase):
    def __init__(self, value: np.number):
        value_type = types.from_numpy_dtype(value.dtype)
        cvalue = types.to_ctypes_type(value_type)(value)
        # state_type is the value itself (not a pointer)
        state_type = value_type
        kind = CountingIteratorKind(value_type, state_type)
        super().__init__(
            kind=kind,
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


class DiscardIteratorKind(IteratorKind):
    pass


class DiscardIterator(IteratorBase):
    def __init__(self, reference_iterator=None):
        from .._utils.temp_storage_buffer import TempStorageBuffer

        if reference_iterator is None:
            reference_iterator = TempStorageBuffer(1)

        if hasattr(reference_iterator, "__cuda_array_interface__"):
            iter = RawPointer(
                reference_iterator.__cuda_array_interface__["data"][0],
                types.from_numpy_dtype(get_dtype(reference_iterator)),
                reference_iterator,
            )
        else:
            iter = reference_iterator

        super().__init__(
            kind=DiscardIteratorKind(iter.value_type, iter.state_type),
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


class ReverseIteratorKind(IteratorKind):
    pass


def make_reverse_iterator(it: DeviceArrayLike | IteratorBase):
    if not hasattr(it, "__cuda_array_interface__") and not isinstance(it, IteratorBase):
        raise NotImplementedError(
            f"Reverse iterator is not implemented for type {type(it)}"
        )

    if hasattr(it, "__cuda_array_interface__"):
        last_element_ptr = _get_last_element_ptr(it)
        it = RawPointer(
            last_element_ptr,
            types.from_numpy_dtype(get_dtype(it)),
            it,
        )

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
        def __init__(self, it):
            self._it = it
            kind = ReverseIteratorKind(it.value_type, it.state_type)
            super().__init__(
                kind=kind,
                cvalue=it.cvalue,
                state_type=it.state_type,
                value_type=it.value_type,
            )

        @property
        def host_advance(self):
            return self._advance

        @property
        def children(self):
            return (self._it,)

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


class TransformIteratorKind(IteratorKind):
    def __init__(
        self, underlying_it_kind: IteratorKind, op: CachableFunction, io_kind: str
    ):
        self.underlying_it_kind = underlying_it_kind
        self.op = op
        self.io_kind = io_kind

    def __repr__(self):
        return f"TransformIteratorKind({self.underlying_it_kind}, {self.op})"

    def __eq__(self, other):
        return (
            type(self) is type(other)
            and self.underlying_it_kind == other.underlying_it_kind
            and self.op == other.op
        )

    def __hash__(self):
        return hash((type(self), self.underlying_it_kind, self.op))


def make_transform_iterator(it: IteratorBase, op: Callable, io_kind: str):
    if hasattr(it, "__cuda_array_interface__"):
        it = pointer(it)

    it_host_advance = it.host_advance
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

    alloca_temp_for_underlying_type = None
    if io_kind == "input":
        from .._intrinsics import make_alloca_intrinsic

        # Create alloca intrinsic (accepts TypeDescriptor directly)
        alloca_temp_for_underlying_type = make_alloca_intrinsic(it.value_type)

    op = cuda.jit(op, device=True)

    class TransformIterator(IteratorBase):
        def __init__(self, it: IteratorBase, op, io_kind: str):
            self._it = it
            self._op = CachableFunction(op.py_func)
            state_type = it.state_type

            kind = TransformIteratorKind(it.kind, self._op, io_kind)
            super().__init__(
                kind=kind,
                cvalue=it.cvalue,
                state_type=state_type,
                value_type=it.value_type,
            )

        @property
        def host_advance(self):
            return it_host_advance

        @property
        def children(self):
            return (self._it,)

        @property
        def advance(self):
            return self._advance

        @property
        def input_dereference(self):
            if it_input_dereference is None:
                raise AttributeError("This iterator is not an input iterator")
            return TransformIterator._input_dereference

        @property
        def output_dereference(self):
            if it_output_dereference is None:
                raise AttributeError("This iterator is not an output iterator")
            return TransformIterator._output_dereference

        @staticmethod
        def _advance(state, distance):
            return it_advance(state, distance)

        @staticmethod
        def _input_dereference(state, result):
            # Allocate temporary storage for the deref input type
            temp_ptr = alloca_temp_for_underlying_type()
            # Call underlying iterator's dereference with temp storage
            it_input_dereference(state, temp_ptr)
            # Apply transformation and store in result
            result[0] = op(temp_ptr[0])

        @staticmethod
        def _output_dereference(state, x):
            it_output_dereference(state, op(x))

        def __add__(self, offset: int):
            res = type(self).__new__(type(self))
            res._it = self._it + offset
            res._op = self._op
            res.state_type = self.state_type
            res.value_type = self.value_type
            res._kind = self._kind
            res.cvalue = type(self.cvalue)(self.cvalue.value + offset)
            res._state = IteratorState(res.cvalue)
            return res

    return TransformIterator(it, op, io_kind)


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


cache_with_registered_key_functions.register(IteratorBase, lambda it: it.kind)
