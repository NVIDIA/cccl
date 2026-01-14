# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Numba-based iterator base classes.

This module contains the IteratorBase class and IteratorKind that form the
foundation for all Numba-based iterator implementations.
"""

import operator
import uuid
from functools import lru_cache
from typing import Tuple

import numba
from llvmlite import ir
from numba import cuda, types
from numba.core.extending import intrinsic, overload

from ..._bindings import IteratorState
from ..interop import make_host_cfunc

_DEVICE_POINTER_SIZE = 8
_DEVICE_POINTER_BITWIDTH = _DEVICE_POINTER_SIZE * 8


@lru_cache(maxsize=256)  # TODO: what's a reasonable value?
def cached_compile(func, sig, abi_name=None, **kwargs):
    return cuda.compile(func, sig, abi_info={"abi_name": abi_name}, **kwargs)


class IteratorKind:
    """
    The `.kind` of an iterator encapsulates additional metadata about the iterator,
    analogous to the `.dtype` of a NumPy array.
    """

    def __init__(self, value_type, state_type):
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


def _get_abi_suffix(kind: IteratorKind):
    return uuid.uuid4().hex


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

    iterator_kind_type: type  # must be a subclass of IteratorKind

    def __init__(
        self,
        cvalue,
        state_type: types.Type,
        value_type: types.Type,
    ):
        """
        Parameters
        ----------
        cvalue
          A ctypes type representing the object pointed to by the iterator.
        state_type
          A numba type representing the type of the input to the advance
          and dereference functions. This should be a pointer type.
        value_type
          The numba type of the value returned by the dereference operation.
        """
        self.cvalue = cvalue
        self.state_type = state_type
        self.state_ptr_type = types.CPointer(state_type)
        self.value_type = value_type

        self._kind = self.__class__.iterator_kind_type(self.value_type, self.state_type)
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
    def is_input_iterator(self) -> bool:
        return self.input_dereference is not None

    @property
    def is_output_iterator(self) -> bool:
        return self.output_dereference is not None

    def get_advance_ltoir(self) -> Tuple:
        from ..odr_helpers import create_advance_void_ptr_wrapper

        abi_name = f"advance_{_get_abi_suffix(self.kind)}"
        wrapped_advance, wrapper_sig = create_advance_void_ptr_wrapper(
            self.advance, self.state_ptr_type
        )
        ltoir, _ = cached_compile(
            wrapped_advance,
            wrapper_sig,
            output="ltoir",
            abi_name=abi_name,
        )
        return (abi_name, ltoir)

    def get_input_dereference_ltoir(self) -> Tuple:
        from ..odr_helpers import create_input_dereference_void_ptr_wrapper

        abi_name = f"input_dereference_{_get_abi_suffix(self.kind)}"
        wrapped_deref, wrapper_sig = create_input_dereference_void_ptr_wrapper(
            self.input_dereference, self.state_ptr_type, self.value_type
        )
        ltoir, _ = cached_compile(
            wrapped_deref,
            wrapper_sig,
            output="ltoir",
            abi_name=abi_name,
        )
        return (abi_name, ltoir)

    def get_output_dereference_ltoir(self) -> Tuple:
        from ..odr_helpers import create_output_dereference_void_ptr_wrapper

        abi_name = f"output_dereference_{_get_abi_suffix(self.kind)}"
        wrapped_deref, wrapper_sig = create_output_dereference_void_ptr_wrapper(
            self.output_dereference, self.state_ptr_type, self.value_type
        )
        ltoir, _ = cached_compile(
            wrapped_deref,
            wrapper_sig,
            output="ltoir",
            abi_name=abi_name,
        )
        return (abi_name, ltoir)

    def to_cccl_iter(self, is_output: bool = False):
        """Convert this iterator to a CCCL Iterator object.

        Args:
            is_output: If True, use output_dereference; otherwise use input_dereference

        Returns:
            CCCL Iterator object for C++ interop
        """
        from numba import cuda

        from ..._bindings import Iterator, Op
        from ..._bindings import IteratorKind as CCCLIteratorKind
        from ...op import OpKind
        from ..interop import numba_type_to_info

        context = cuda.descriptor.cuda_target.target_context
        state_ptr_type = self.state_ptr_type
        state_type = self.state_type
        size = context.get_value_type(state_type).get_abi_size(context.target_data)
        iterator_state = memoryview(self.state)
        if not iterator_state.nbytes == size:
            raise ValueError(
                f"Iterator state size, {iterator_state.nbytes} bytes, for iterator type {type(self)} "
                f"does not match size of numba type, {size} bytes"
            )
        alignment = context.get_value_type(state_ptr_type).get_abi_alignment(
            context.target_data
        )

        advance_abi_name, advance_ltoir = self.get_advance_ltoir()
        if is_output:
            deref_abi_name, deref_ltoir = self.get_output_dereference_ltoir()
        else:
            deref_abi_name, deref_ltoir = self.get_input_dereference_ltoir()

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

        if self.host_advance is not None:
            try:
                host_advance_fn = make_host_cfunc(
                    self.state_ptr_type, self.host_advance
                )
            except Exception:
                # TODO: figure out what to do here.
                host_advance_fn = None
        else:
            host_advance_fn = None

        return Iterator(
            alignment,
            CCCLIteratorKind.ITERATOR,
            advance_op,
            deref_op,
            numba_type_to_info(self.value_type),
            state=self.state,
            host_advance_fn=host_advance_fn,
        )

    def __add__(self, offset: int):
        # add the offset to the iterator's state, and return a new iterator
        # with the new state.
        res = type(self).__new__(type(self))
        res.state_ptr_type = self.state_ptr_type
        res.state_type = self.state_type
        res.value_type = self.value_type
        res._kind = self._kind
        res.cvalue = type(self.cvalue)(self.cvalue.value + offset)
        res._state = IteratorState(res.cvalue)

        return res


# =============================================================================
# Pointer arithmetic intrinsics
# =============================================================================


def sizeof_pointee(context, ptr):
    size = context.get_abi_sizeof(ptr.type.pointee)
    return ir.Constant(ir.IntType(_DEVICE_POINTER_BITWIDTH), size)


@intrinsic
def pointer_add_intrinsic(context, ptr, offset):
    def codegen(context, builder, sig, args):
        ptr, index = args
        base = builder.ptrtoint(ptr, ir.IntType(_DEVICE_POINTER_BITWIDTH))
        sizeof = sizeof_pointee(context, ptr)
        # Cast index to match sizeof type if needed
        if index.type != sizeof.type:
            index = (
                builder.sext(index, sizeof.type)
                if index.type.width < sizeof.type.width
                else builder.trunc(index, sizeof.type)
            )
        offset = builder.mul(index, sizeof)
        result = builder.add(base, offset)
        return builder.inttoptr(result, ptr.type)

    return ptr(ptr, offset), codegen


@overload(operator.add)
def pointer_add(ptr, offset):
    if not isinstance(ptr, numba.types.CPointer) or not isinstance(
        offset, numba.types.Integer
    ):
        return

    def impl(ptr, offset):
        return pointer_add_intrinsic(ptr, offset)

    return impl
