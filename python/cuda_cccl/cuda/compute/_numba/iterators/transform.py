# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
TransformIterator implementation.

TransformIterator wraps another iterator and applies a transformation
function to each element on dereference.
"""

from typing import Callable

import numba
from numba import cuda, types
from numba.core.extending import intrinsic
from numba.cuda.dispatcher import CUDADispatcher

from .._caching import CachableFunction
from ..numba_utils import get_inferred_return_type, signature_from_annotations
from .base import IteratorBase, IteratorKind
from .simple import pointer


class TransformIteratorKind(IteratorKind):
    pass


def make_transform_iterator(it, op: Callable, io_kind: str):
    if hasattr(it, "__cuda_array_interface__"):
        it = pointer(it, numba.from_dtype(it.dtype))

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

    if io_kind == "input":
        underlying_it_type = it.value_type

    op = cuda.jit(op, device=True)

    # Create a specialized intrinsic for allocating temp storage of the underlying type
    @intrinsic
    def alloca_temp_for_underlying_type(context):
        def codegen(context, builder, sig, args):
            temp_value_type = context.get_value_type(underlying_it_type)
            temp_ptr = builder.alloca(temp_value_type)
            return temp_ptr

        return types.CPointer(underlying_it_type)(), codegen

    class TransformIterator(IteratorBase):
        iterator_kind_type = TransformIteratorKind

        def __init__(self, it: IteratorBase, op: CUDADispatcher, io_kind: str):
            self._it = it
            self._op = CachableFunction(op.py_func)
            state_type = it.state_type

            if io_kind == "input":
                # For input iterators, use annotations if available, otherwise
                # rely on numba to infer the return type.
                try:
                    value_type = signature_from_annotations(op.py_func).args[0]
                except ValueError:
                    value_type = get_inferred_return_type(
                        op.py_func, (underlying_it_type,)
                    )
            else:
                # For output iterators, always require annotations.
                # The inferred type may not match the iterator being
                # written to.
                value_type = signature_from_annotations(op.py_func).args[0]

            super().__init__(
                cvalue=it.cvalue,
                state_type=state_type,
                value_type=value_type,
            )
            self._kind = self.__class__.iterator_kind_type(
                (self._it.value_type, self._it.kind, self._op), self.state_type
            )

        @property
        def host_advance(self):
            return it_host_advance

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

    return TransformIterator(it, op, io_kind)
