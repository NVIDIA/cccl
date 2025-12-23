# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ctypes

from llvmlite import ir  # noqa: F401
from numba import cuda, types  # noqa: F401
from numba.core.datamodel.registry import default_manager  # noqa: F401
from numba.core.extending import as_numba_type, intrinsic  # noqa: F401

from .._caching import cache_with_key
from ..struct import make_struct_type
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
    ZipState = make_struct_type("ZipState", state_field_names, state_field_types)

    # this iterator's value is a struct composed of the values of the input iterators:
    value_field_names = tuple(f"value_{i}" for i in range(n_iterators))
    value_field_types = tuple(it.value_type for it in iterators)
    ZipValue = make_struct_type("ZipValue", value_field_names, value_field_types)

    cvalue = ZipCValueStruct(
        **{f"iter_{i}": it.cvalue for i, it in enumerate(iterators)}
    )
    state_type = as_numba_type(ZipState)
    value_type = as_numba_type(ZipValue)
    return cvalue, state_type, value_type, ZipValue


def _make_cache_key(iterators):
    return tuple(
        it.kind if isinstance(it, IteratorBase) else it.dtype for it in iterators
    )


@cache_with_key(_make_cache_key)
def _get_advance_and_dereference_functions(iterators):
    # Generate the advance and dereference functions for the zip iterator
    # composed of the input iterators
    # Put simply, the advance method invokes the advance method of each input
    # iterator, and the dereference method invokes the dereference method of each
    # input iterator.

    n_iterators = len(iterators)

    # Create a local namespace for this zip iterator to avoid polluting globals
    # and prevent name collisions when nesting zip iterators
    local_ns = {
        "intrinsic": intrinsic,
        "types": types,
        "ir": ir,
        "default_manager": default_manager,
    }

    # Within the advance and dereference methods of this iterator, we need a way
    # to get pointers to the fields of the state struct (advance and dereference),
    # and the value struct (dereference). This needs `n` custom intrinsics, one
    # for each field. The loop below defines those intrinsics:
    for field_idx in range(n_iterators):
        func_name = f"get_field_ptr_{field_idx}"

        # Create the intrinsic function
        intrinsic_code = f"""
@intrinsic
def {func_name}(context, struct_ptr_type):
    def codegen(context, builder, sig, args):
        struct_ptr = args[0]
        # Use GEP to get pointer to field at index {field_idx}
        field_ptr = builder.gep(
            struct_ptr,
            [ir.Constant(ir.IntType(32), 0), ir.Constant(
                ir.IntType(32), {field_idx})],
        )
        return field_ptr

    struct_model = default_manager.lookup(struct_ptr_type.dtype)
    field_type = struct_model._members[{field_idx}]
    return types.CPointer(field_type)(struct_ptr_type), codegen
"""
        # Execute the code to create the intrinsic function in local namespace
        exec(intrinsic_code, local_ns)

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
    from .._cccl_interop import get_value_type

    if len(iterators) < 1:
        raise ValueError("At least 1 iterator is required")

    # Convert arrays to iterators if needed
    processed_iterators = []
    for it in iterators:
        if hasattr(it, "__cuda_array_interface__"):
            it = pointer(it, get_value_type(it))
        processed_iterators.append(it)

    # Validate all are iterators
    for i, it in enumerate(processed_iterators):
        if not isinstance(it, IteratorBase):
            raise TypeError(f"Argument {i} must be an iterator or device array")

    cvalue, state_type, value_type, ZipValue = _get_zip_iterator_metadata(
        processed_iterators
    )

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
            self._kind = self.__class__.iterator_kind_type(
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
            return input_dereference_func

        @property
        def output_dereference(self):
            if not all_support_output:
                raise AttributeError(
                    "ZipIterator cannot be used as output iterator when not all "
                    "underlying iterators support output"
                )
            return output_dereference_func

    return ZipIterator(processed_iterators)
