# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ctypes
import uuid

import numba
from llvmlite import ir  # noqa: F401
from numba import cuda, types  # noqa: F401
from numba.core.datamodel.registry import default_manager  # noqa: F401
from numba.core.extending import intrinsic  # noqa: F401

from .._utils.protocols import get_dtype
from ..struct import gpu_struct_from_numba_types
from ._iterators import (
    IteratorBase,
    IteratorKind,
    pointer,
)


class ZipIteratorKind(IteratorKind):
    pass


def _get_zip_iterator_metadata(iterators):
    # return the cvalue, state_type, and value_type for the zip iterator
    # composed of the input iterators

    n_iterators = len(iterators)

    # ctypes struct for the combined state:
    fields = [(f"iter_{i}", it.cvalue.__class__) for i, it in enumerate(iterators)]
    ZipCValueStruct = type("ZipCValueStruct", (ctypes.Structure,), {"_fields_": fields})

    # this iterator's state is a struct composed of the states of the input iterators:
    state_field_names = tuple(f"state_{i}" for i in range(n_iterators))
    state_field_types = tuple(it.state_type for it in iterators)
    ZipState = gpu_struct_from_numba_types(
        "ZipState", state_field_names, state_field_types
    )

    # this iterator's value is a struct composed of the values of the input iterators:
    value_field_names = tuple(f"value_{i}" for i in range(n_iterators))
    value_field_types = tuple(it.value_type for it in iterators)
    ZipValue = gpu_struct_from_numba_types(
        "ZipValue", value_field_names, value_field_types
    )

    cvalue = ZipCValueStruct(
        **{f"iter_{i}": it.cvalue for i, it in enumerate(iterators)}
    )
    state_type = ZipState._numba_type
    value_type = ZipValue._numba_type
    return cvalue, state_type, value_type


def _get_advance_and_dereference_functions(
    iterators, state_field_types, value_field_types
):
    # Generate the advance and dereference functions for the zip iterator
    # composed of the input iterators
    # Put simply, the advance method invokes the advance method of each input
    # iterator, and the dereference method invokes the dereference method of each
    # input iterator.

    n_iterators = len(iterators)

    # Generate a unique ID for this zip iterator to avoid name collisions
    # when nesting zip iterators
    unique_id = uuid.uuid4().hex[:8]

    # Within the advance and dereference methods of this iterator, we need a way
    # to get pointers to the fields of the state struct (advance and dereference),
    # and the value struct (dereference). This needs `n` custom intrinsics, one
    # for each field. We create separate intrinsics for state and value fields.

    # Create intrinsics for state field pointers
    for field_idx in range(n_iterators):
        func_name_state = f"get_state_field_ptr_{field_idx}_{unique_id}"
        func_name_value = f"get_value_field_ptr_{field_idx}_{unique_id}"

        # Create the intrinsic function for state fields
        intrinsic_code_state = f"""
@intrinsic
def {func_name_state}(context, struct_ptr_type):
    def codegen(context, builder, sig, args):
        struct_ptr = args[0]
        # Use GEP to get pointer to field at index {field_idx}
        field_ptr = builder.gep(
            struct_ptr,
            [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), {field_idx})],
        )
        return field_ptr

    # Use the pre-computed field type
    field_type = state_field_types[{field_idx}]
    return types.CPointer(field_type)(struct_ptr_type), codegen
"""

        # Create the intrinsic function for value fields
        intrinsic_code_value = f"""
@intrinsic
def {func_name_value}(context, struct_ptr_type):
    def codegen(context, builder, sig, args):
        struct_ptr = args[0]
        # Use GEP to get pointer to field at index {field_idx}
        field_ptr = builder.gep(
            struct_ptr,
            [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), {field_idx})],
        )
        return field_ptr

    # Use the pre-computed field type
    field_type = value_field_types[{field_idx}]
    return types.CPointer(field_type)(struct_ptr_type), codegen
"""
        # Make field types available in the exec context
        exec_globals = globals().copy()
        exec_globals["state_field_types"] = state_field_types
        exec_globals["value_field_types"] = value_field_types

        # Execute the code to create the intrinsic functions
        exec(intrinsic_code_state, exec_globals)
        exec(intrinsic_code_value, exec_globals)

        # Copy back to globals
        globals()[func_name_state] = exec_globals[func_name_state]
        globals()[func_name_value] = exec_globals[func_name_value]

    # Now we can define the advance and dereference methods of this iterator,
    # which also need to be defined dynamically because they will use the
    # intrinsics defined above.
    for i, it in enumerate(iterators):
        globals()[f"advance_{i}_{unique_id}"] = cuda.jit(it.advance, device=True)
        globals()[f"dereference_{i}_{unique_id}"] = cuda.jit(
            it.input_dereference, device=True
        )  # only input_dereference for now

    # Generate the advance method, which advances each input iterator:
    advance_lines = []  # lines of code for the advance method
    dereference_lines = []  # lines of code for the dereference method
    for i in range(n_iterators):
        advance_lines.append(
            f"    state_ptr_{i} = get_state_field_ptr_{i}_{unique_id}(state)\n"
            f"    advance_{i}_{unique_id}(state_ptr_{i}, distance)"
        )
        dereference_lines.append(
            f"    state_ptr_{i} = get_state_field_ptr_{i}_{unique_id}(state)\n"
            f"    result_ptr_{i} = get_value_field_ptr_{i}_{unique_id}(result)\n"
            f"    dereference_{i}_{unique_id}(state_ptr_{i}, result_ptr_{i})"
        )

    advance_method_code = f"""
def input_advance_{unique_id}(state, distance):
    # Advance each iterator using dynamically created field pointer functions
{chr(10).join(advance_lines)}
"""  # chr(10) is '\n'

    dereference_method_code = f"""
def input_dereference_{unique_id}(state, result):
    # Dereference each iterator using dynamically created field pointer functions
{chr(10).join(dereference_lines)}
"""  # chr(10) is '\n'

    # Execute the method codes:
    exec(advance_method_code, globals())
    exec(dereference_method_code, globals())

    advance_func = globals()[f"input_advance_{unique_id}"]
    dereference_func = globals()[f"input_dereference_{unique_id}"]

    return advance_func, dereference_func


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
            it = pointer(it, numba.from_dtype(get_dtype(it)))
        processed_iterators.append(it)

    # Validate all are iterators
    for i, it in enumerate(processed_iterators):
        if not isinstance(it, IteratorBase):
            raise TypeError(f"Argument {i} must be an iterator or device array")

    cvalue, state_type, value_type = _get_zip_iterator_metadata(processed_iterators)

    # Extract field types for the intrinsics
    state_field_types = tuple(it.state_type for it in processed_iterators)
    value_field_types = tuple(it.value_type for it in processed_iterators)

    advance_func, dereference_func = _get_advance_and_dereference_functions(
        processed_iterators, state_field_types, value_field_types
    )

    class ZipIterator(IteratorBase):
        iterator_kind_type = ZipIteratorKind

        def __init__(self, iterators_list):
            self._iterators = iterators_list
            self._n_iterators = len(iterators_list)

            kinds = [it.kind for it in iterators_list]
            super().__init__(
                cvalue=cvalue,
                state_type=state_type,
                value_type=value_type,
            )
            self.kind_ = self.__class__.iterator_kind_type(
                (value_type, *kinds), self.state_type
            )

        @property
        def host_advance(self):
            return advance_func

        @property
        def advance(self):
            return advance_func

        @property
        def input_dereference(self):
            return dereference_func

    return ZipIterator(processed_iterators)
