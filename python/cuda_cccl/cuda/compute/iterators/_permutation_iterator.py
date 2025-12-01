# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ctypes

import numba
from llvmlite import ir  # noqa: F401
from numba import cuda, types  # noqa: F401
from numba.core.datamodel.registry import default_manager  # noqa: F401
from numba.core.extending import as_numba_type, intrinsic  # noqa: F401

from .._caching import cache_with_key
from .._cccl_interop import get_dtype
from ..struct import make_struct_type
from ._iterators import (
    IteratorBase,
    IteratorKind,
    pointer,
)


class PermutationIteratorKind(IteratorKind):
    pass


def _make_cache_key(values, indices):
    """Create a cache key based on value type and iterator kinds."""
    return (values.value_type, values.kind, indices.kind)


@cache_with_key(_make_cache_key)
def _generate_advance_and_dereference_methods(values, indices):
    values_state_type = values.state_type
    index_type = indices.value_type

    # Check if values supports output operations
    values_is_output_iterator = False
    try:
        output_deref = values.output_dereference
        if output_deref is not None:
            values_is_output_iterator = True
    except AttributeError:
        pass

    # Create a local namespace to avoid polluting globals
    local_ns = {
        "intrinsic": intrinsic,
        "types": types,
        "ir": ir,
        "default_manager": default_manager,
        "cuda": cuda,
        "index_type": index_type,
        "values_state_type": values_state_type,
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

    # Define intrinsics for accessing struct fields
    intrinsic_code = """
@intrinsic
def get_value_state_field_ptr(context, struct_ptr_type):
    def codegen(context, builder, sig, args):
        struct_ptr = args[0]
        # Use GEP to get pointer to field at index 0 (value_state)
        field_ptr = builder.gep(
            struct_ptr,
            [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 0)],
        )
        return field_ptr

    struct_model = default_manager.lookup(struct_ptr_type.dtype)
    field_type = struct_model._members[0]
    return types.CPointer(field_type)(struct_ptr_type), codegen

@intrinsic
def get_index_state_field_ptr(context, struct_ptr_type):
    def codegen(context, builder, sig, args):
        struct_ptr = args[0]
        # Use GEP to get pointer to field at index 1 (index_state)
        field_ptr = builder.gep(
            struct_ptr,
            [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 1)],
        )
        return field_ptr

    struct_model = default_manager.lookup(struct_ptr_type.dtype)
    field_type = struct_model._members[1]
    return types.CPointer(field_type)(struct_ptr_type), codegen

@intrinsic
def alloca_temp_for_index_type(context):
    def codegen(context, builder, sig, args):
        temp_value_type = context.get_value_type(index_type)
        temp_ptr = builder.alloca(temp_value_type)
        return temp_ptr

    return types.CPointer(index_type)(), codegen

@intrinsic
def alloca_temp_for_value_state(context):
    def codegen(context, builder, sig, args):
        temp_state_type = context.get_value_type(values_state_type)
        temp_ptr = builder.alloca(temp_state_type)
        return temp_ptr

    return types.CPointer(values_state_type)(), codegen
"""

    # Execute intrinsic definitions in local namespace
    exec(intrinsic_code, local_ns)

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
        values = pointer(values, numba.from_dtype(get_dtype(values)))
    elif not isinstance(values, IteratorBase):
        raise TypeError("values must be a device array or iterator")

    if hasattr(indices, "__cuda_array_interface__"):
        indices = pointer(indices, numba.from_dtype(get_dtype(indices)))
    elif not isinstance(indices, IteratorBase):
        raise TypeError("indices must be an iterator or device array")

    # Get the cached advance and dereference methods
    advance_func, input_dereference_func, output_dereference_func = (
        _generate_advance_and_dereference_methods(values, indices)
    )

    # The cvalue and state for PermutationIterator are
    # structs composed of the cvalues and states of the
    # value and index iterators.
    class PermutationCValueStruct(ctypes.Structure):
        _fields_ = [
            ("value_state", values.cvalue.__class__),
            ("index_state", indices.cvalue.__class__),
        ]

    PermutationState = make_struct_type(
        "PermutationState",
        field_names=("value_state", "index_state"),
        field_types=(values.state_type, indices.state_type),
    )

    cvalue = PermutationCValueStruct(values.cvalue, indices.cvalue)
    state_type = as_numba_type(PermutationState)
    value_type = values.value_type

    class PermutationIterator(IteratorBase):
        iterator_kind_type = PermutationIteratorKind

        def __init__(self, values_it, indices_it):
            self._values = values_it
            self._indices = indices_it
            super().__init__(
                cvalue=cvalue,
                state_type=state_type,
                value_type=value_type,
            )
            self._kind = self.__class__.iterator_kind_type(
                (value_type, values_it.kind, indices_it.kind), state_type
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

    return PermutationIterator(values, indices)
