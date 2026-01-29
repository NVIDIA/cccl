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


class ZipIteratorKind(IteratorKind):
    def __init__(
        self,
        value_type: types.TypeDescriptor,
        iterators_kinds: tuple[IteratorKind, ...],
    ):
        self.iterators_kinds = iterators_kinds
        self.value_type = value_type

    def __repr__(self):
        return f"ZipIteratorKind({self.value_type}, {self.iterators_kinds})"

    def __eq__(self, other):
        return (
            type(self) is type(other)
            and self.value_type == other.value_type
            and self.iterators_kinds == other.iterators_kinds
        )

    def __hash__(self):
        return hash((type(self), self.value_type, self.iterators_kinds))


def _get_zip_iterator_metadata(iterators):
    from ..struct import gpu_struct

    # Create ctypes struct for cvalue
    fields = [(f"iter_{i}", it.cvalue.__class__) for i, it in enumerate(iterators)]
    ZipCValueStruct = type("ZipCValueStruct", (ctypes.Structure,), {"_fields_": fields})

    # Create state struct using gpu_struct
    state_fields = {f"state_{i}": it.state_type for i, it in enumerate(iterators)}
    ZipState = gpu_struct(state_fields, name="ZipState")
    state_type = ZipState._type_descriptor

    # Create value struct using gpu_struct
    value_fields = {f"value_{i}": it.value_type for i, it in enumerate(iterators)}
    ZipValue = gpu_struct(value_fields, name="ZipValue")
    value_type = ZipValue._type_descriptor

    cvalue = ZipCValueStruct(
        **{f"iter_{i}": it.cvalue for i, it in enumerate(iterators)}
    )
    return cvalue, state_type, value_type


# Automatic: iterators tuple → (kind, kind, ...) or (dtype, dtype, ...)
@cache_with_registered_key_functions
def _get_advance_and_dereference_functions(iterators):
    # Generate the advance and dereference functions for the zip iterator
    # composed of the input iterators
    # Put simply, the advance method invokes the advance method of each input
    # iterator, and the dereference method invokes the dereference method of each
    # input iterator.

    from .._intrinsics import make_struct_field_ptr_intrinsic

    n_iterators = len(iterators)

    # Create a local namespace for this zip iterator to avoid polluting globals
    # and prevent name collisions when nesting zip iterators
    local_ns = {}

    # Within the advance and dereference methods of this iterator, we need a way
    # to get pointers to the fields of the state struct (advance and dereference),
    # and the value struct (dereference). This needs `n` intrinsics, one
    # for each field. Create them using the factory:
    for field_idx in range(n_iterators):
        local_ns[f"get_field_ptr_{field_idx}"] = make_struct_field_ptr_intrinsic(
            field_idx
        )

    # Now we can define the advance and dereference methods of this iterator,
    # which also need to be defined dynamically because they will use the
    # intrinsics defined above.
    for i, it in enumerate(iterators):
        local_ns[f"advance_{i}"] = cuda.jit(it.advance, device=True)
        local_ns[f"input_dereference_{i}"] = cuda.jit(it.input_dereference, device=True)
        # Also compile output_dereference if available
        try:
            output_deref = it.output_dereference
            if output_deref is not None:
                local_ns[f"output_dereference_{i}"] = cuda.jit(
                    output_deref, device=True
                )
        except AttributeError:
            # Iterator doesn't support output operations
            pass

    # Generate the advance method, which advances each input iterator:
    advance_lines = []  # lines of code for the advance method
    input_dereference_lines = []  # lines of code for input dereference method
    output_dereference_lines = []  # lines of code for output dereference method

    # Check if all iterators support output operations
    def supports_output(it):
        try:
            return it.output_dereference is not None
        except AttributeError:
            return False

    all_support_output = all(supports_output(it) for it in iterators)

    for i in range(n_iterators):
        advance_lines.append(
            f"    state_ptr_{i} = get_field_ptr_{i}(state)\n"
            f"    advance_{i}(state_ptr_{i}, distance)"
        )
        input_dereference_lines.append(
            f"    state_ptr_{i} = get_field_ptr_{i}(state)\n"
            f"    result_ptr_{i} = get_field_ptr_{i}(result)\n"
            f"    input_dereference_{i}(state_ptr_{i}, result_ptr_{i})"
        )
        if all_support_output:
            output_dereference_lines.append(
                f"    state_ptr_{i} = get_field_ptr_{i}(state)\n"
                f"    output_dereference_{i}(state_ptr_{i}, x.value_{i})"
            )

    advance_method_code = f"""
def input_advance(state, distance):
    # Advance each iterator using dynamically created field pointer functions
{chr(10).join(advance_lines)}
"""  # chr(10) is '\n'

    input_dereference_method_code = f"""
def input_dereference(state, result):
    # Dereference each iterator using dynamically created field pointer functions
{chr(10).join(input_dereference_lines)}
"""  # chr(10) is '\n'

    # Execute the method codes in local namespace:
    exec(advance_method_code, local_ns)
    exec(input_dereference_method_code, local_ns)

    advance_func = local_ns["input_advance"]
    input_dereference_func = local_ns["input_dereference"]

    # Generate output_dereference if all iterators support it
    output_dereference_func = None
    if all_support_output:
        output_dereference_method_code = f"""
def output_dereference(state, x):
    # Write to each iterator using dynamically created field pointer functions
{chr(10).join(output_dereference_lines)}
"""
        exec(output_dereference_method_code, local_ns)
        output_dereference_func = local_ns["output_dereference"]

    return advance_func, input_dereference_func, output_dereference_func


def make_zip_iterator(*iterators):
    """
    Create a ZipIterator that combines N iterators.

    Args:
        *iterators: Variable number of iterators or device arrays

    Returns:
        ZipIterator: Iterator that combines all input iterators
    """
    if len(iterators) < 1:
        raise ValueError("At least 1 iterator is required")

    # Convert arrays to iterators if needed
    processed_iterators = []
    for it in iterators:
        if hasattr(it, "__cuda_array_interface__"):
            it = pointer(it)
        processed_iterators.append(it)

    # Validate all are iterators
    for i, it in enumerate(processed_iterators):
        if not isinstance(it, IteratorBase):
            raise TypeError(f"Argument {i} must be an iterator or device array")

    cvalue, state_type, value_type = _get_zip_iterator_metadata(processed_iterators)

    advance_func, input_dereference_func, output_dereference_func = (
        _get_advance_and_dereference_functions(processed_iterators)
    )

    # Check if all underlying iterators support output
    def supports_output(it):
        try:
            return it.output_dereference is not None
        except AttributeError:
            return False

    all_support_output = all(supports_output(it) for it in processed_iterators)

    class ZipIterator(IteratorBase):
        def __init__(self, iterators_list):
            self._iterators = iterators_list
            self._n_iterators = len(iterators_list)

            kinds = [it.kind for it in iterators_list]
            kind = ZipIteratorKind(value_type, tuple(kinds))
            super().__init__(
                kind=kind,
                cvalue=cvalue,
                state_type=state_type,
                value_type=value_type,
            )

        @property
        def host_advance(self):
            return advance_func

        @property
        def advance(self):
            return advance_func

        @property
        def input_dereference(self):
            return input_dereference_func

        @property
        def output_dereference(self):
            if not all_support_output:
                raise AttributeError(
                    "ZipIterator cannot be used as output iterator when not all "
                    "underlying iterators support output"
                )
            return output_dereference_func

        @property
        def children(self):
            return tuple(self._iterators)

        def _rebuild_value_type_from_children(self):
            from ..struct import gpu_struct

            if not self._iterators:
                raise ValueError("Zip iterator has no children")
            value_fields = {
                f"value_{i}": child.value_type
                for i, child in enumerate(self._iterators)
            }
            self.value_type = gpu_struct(value_fields, name="ZipValue")._type_descriptor

    return ZipIterator(processed_iterators)
