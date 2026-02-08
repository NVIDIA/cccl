# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ctypes

from numba import cuda

from .. import types
from .._caching import cache_with_registered_key_functions
from ._iterators import (
    IteratorBase,
    IteratorKind,
    pointer,
)


class PermutationIteratorKind(IteratorKind):
    def __init__(
        self,
        value_type: types.TypeDescriptor,
        values_kind: IteratorKind,
        indices_kind: IteratorKind,
    ):
        self.values_kind = values_kind
        self.indices_kind = indices_kind

    def __repr__(self):
        return f"PermutationIteratorKind({self.value_type}, {self.values_kind}, {self.indices_kind})"

    def __eq__(self, other):
        return (
            type(self) is type(other)
            and self.values_kind == other.values_kind
            and self.indices_kind == other.indices_kind
        )

    def __hash__(self):
        return hash((type(self), self.values_kind, self.indices_kind))


@cache_with_registered_key_functions
def _generate_advance_and_dereference_methods(values, indices):
    from .._intrinsics import make_alloca_intrinsic, make_struct_field_ptr_intrinsic

    # Check if values supports output operations
    values_is_output_iterator = False
    try:
        output_deref = values.output_dereference
        if output_deref is not None:
            values_is_output_iterator = True
    except AttributeError:
        pass

    alloca_temp_for_index_type = make_alloca_intrinsic(indices.value_type)
    alloca_temp_for_value_state = make_alloca_intrinsic(values.state_type)
    get_value_state_field_ptr = make_struct_field_ptr_intrinsic(0)
    get_index_state_field_ptr = make_struct_field_ptr_intrinsic(1)

    # Create a local namespace to avoid polluting globals
    local_ns = {
        "alloca_temp_for_index_type": alloca_temp_for_index_type,
        "alloca_temp_for_value_state": alloca_temp_for_value_state,
        "get_value_state_field_ptr": get_value_state_field_ptr,
        "get_index_state_field_ptr": get_index_state_field_ptr,
    }

    # JIT compile value advance/dereference methods
    local_ns["value_advance"] = cuda.jit(values.advance, device=True)
    local_ns["value_input_dereference"] = cuda.jit(
        values.input_dereference, device=True
    )

    if values_is_output_iterator:
        local_ns["value_output_dereference"] = cuda.jit(
            values.output_dereference, device=True
        )

    # JIT compile index advance/dereference methods
    local_ns["index_advance"] = cuda.jit(indices.advance, device=True)
    local_ns["index_input_dereference"] = cuda.jit(
        indices.input_dereference, device=True
    )

    # Define the advance method
    advance_method_code = """
def permutation_advance(state, distance):
    # advance the index iterator
    index_state_ptr = get_index_state_field_ptr(state)
    index_advance(index_state_ptr, distance)
"""

    # Define the input dereference method
    input_dereference_method_code = """
def permutation_input_dereference(state, result):
    # dereference index to get the index value
    index_state_ptr = get_index_state_field_ptr(state)
    temp_index = alloca_temp_for_index_type()
    index_input_dereference(index_state_ptr, temp_index)

    # copy the value state (which always points to position 0)
    # and advance it by the index value
    value_state_ptr = get_value_state_field_ptr(state)
    temp_value_state = alloca_temp_for_value_state()
    temp_value_state[0] = value_state_ptr[0]
    value_advance(temp_value_state, temp_index[0])
    value_input_dereference(temp_value_state, result)
"""

    # Execute the method definitions in local namespace
    exec(advance_method_code, local_ns)
    exec(input_dereference_method_code, local_ns)

    advance_func = local_ns["permutation_advance"]
    input_dereference_func = local_ns["permutation_input_dereference"]
    output_dereference_func = None

    # Define the output dereference method if values supports it
    if values_is_output_iterator:
        output_dereference_method_code = """
def permutation_output_dereference(state, x):
    # dereference index to get the index value
    index_state_ptr = get_index_state_field_ptr(state)
    temp_index = alloca_temp_for_index_type()
    index_input_dereference(index_state_ptr, temp_index)

    # copy the value state (which always points to position 0)
    # and advance it by the index value
    value_state_ptr = get_value_state_field_ptr(state)
    temp_value_state = alloca_temp_for_value_state()
    temp_value_state[0] = value_state_ptr[0]
    value_advance(temp_value_state, temp_index[0])
    value_output_dereference(temp_value_state, x)
"""
        exec(output_dereference_method_code, local_ns)
        output_dereference_func = local_ns["permutation_output_dereference"]

    return advance_func, input_dereference_func, output_dereference_func


def make_permutation_iterator(values, indices):
    """
    Create a PermutationIterator that accesses values through an index mapping.

    The permutation iterator accesses elements from `values` using indices from `indices`,
    effectively computing values[indices[i]] at position i.

    Args:
        values: The values array or iterator to permute
        indices: The indices array or iterator specifying the permutation

    Returns:
        PermutationIterator: Iterator that yields permuted values
    """

    # Convert arrays to iterators if needed
    if hasattr(values, "__cuda_array_interface__"):
        values = pointer(values)
    elif not isinstance(values, IteratorBase):
        raise TypeError("values must be a device array or iterator")

    if hasattr(indices, "__cuda_array_interface__"):
        indices = pointer(indices)
    elif not isinstance(indices, IteratorBase):
        raise TypeError("indices must be an iterator or device array")

    # Get the cached advance and dereference methods
    advance_func, input_dereference_func, output_dereference_func = (
        _generate_advance_and_dereference_methods(values, indices)
    )

    # The cvalue and state for PermutationIterator are
    # structs composed of the cvalues and states of the
    # value and index iterators.
    from ..struct import gpu_struct

    class PermutationCValueStruct(ctypes.Structure):
        _fields_ = [
            ("value_state", values.cvalue.__class__),
            ("index_state", indices.cvalue.__class__),
        ]

    # Create state struct using gpu_struct
    PermutationState = gpu_struct(
        {"value_state": values.state_type, "index_state": indices.state_type},
        name="PermutationState",
    )
    state_type = PermutationState._type_descriptor

    cvalue = PermutationCValueStruct(values.cvalue, indices.cvalue)
    value_type = values.value_type

    class PermutationIterator(IteratorBase):
        def __init__(self, values_it, indices_it):
            self._values = values_it
            self._indices = indices_it
            kind = PermutationIteratorKind(
                value_type, self._values.kind, self._indices.kind
            )
            super().__init__(
                kind=kind,
                cvalue=cvalue,
                state_type=state_type,
                value_type=value_type,
            )

        @property
        def advance(self):
            return advance_func

        @property
        def input_dereference(self):
            return input_dereference_func

        @property
        def output_dereference(self):
            if output_dereference_func is None:
                raise AttributeError(
                    "PermutationIterator cannot be used as output iterator "
                    "when values iterator does not support output"
                )
            return output_dereference_func

        @property
        def children(self):
            return (self._values, self._indices)

        def _rebuild_value_type_from_children(self):
            self.value_type = self._values.value_type

    return PermutationIterator(values, indices)
