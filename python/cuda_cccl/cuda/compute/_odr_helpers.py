# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
ODR (One Definition Rule) Helpers for CCCL Python Interop.

This module provides utilities to create wrapper functions for
device functions that are defined in Python and JIT compiled by Numba.

On the C++ side, these functions are declared as `extern "C"` functions with
void* parameters - the arguments types can not be known at C++ compile time.

Thus, the helpers in this module generate wrapper device functions that accept
void* arguments (matching C++ declarations), cast them to the correct
typed arguments, load/store values as needed, and call the original
function with properly typed arguments.

Example flow:
    User provides: def add(x: int32, y: int32) -> int32
    Wrapper signature: void(void*, void*, void*)  # x_ptr, y_ptr, result_ptr
    C++ sees: extern "C" void wrapped_add(void*, void*, void*);
"""

from __future__ import annotations

import enum
import textwrap
from typing import TYPE_CHECKING

from numba import types
from numba.core.extending import intrinsic

if TYPE_CHECKING:
    from numba.core.typing import Signature

__all__ = [
    "create_op_void_ptr_wrapper",
    "create_advance_void_ptr_wrapper",
    "create_input_dereference_void_ptr_wrapper",
    "create_output_dereference_void_ptr_wrapper",
]


class _ArgMode(enum.Enum):
    """How a void* argument should be handled in wrapper codegen."""

    LOAD = "load"  # Cast to typed pointer, load value
    PTR = "ptr"  # Cast to typed pointer, pass pointer directly
    STORE = "store"  # Cast to typed pointer, store return value here


class _ArgSpec:
    """Specification for a wrapper argument."""

    __slots__ = ("numba_type", "mode")

    def __init__(self, numba_type, mode: _ArgMode):
        self.numba_type = numba_type
        self.mode = mode


def _codegen_void_ptr_wrapper(
    context, builder, args, arg_specs, func_device, inner_sig
):
    """Generate LLVM IR for a void* wrapper function.

    This is the codegen implementation shared by all void* wrappers.
    It processes each argument according to its _ArgSpec mode, calls
    the inner function, and stores the result if needed.

    Args:
        context: Numba codegen context
        builder: LLVM IR builder
        args: LLVM values for the void* arguments
        arg_specs: List of _ArgSpec describing each argument
        func_device: The device function to call
        inner_sig: Numba signature for the inner function

    Returns:
        LLVM dummy value (for void return)
    """

    input_vals = []
    ret_ptr = None

    for i, (arg, spec) in enumerate(zip(args, arg_specs)):
        match spec.mode:
            case _ArgMode.LOAD:
                # Cast void* to typed pointer and load value
                llvm_type = context.get_value_type(spec.numba_type)
                typed_ptr = builder.bitcast(arg, llvm_type.as_pointer())
                val = builder.load(typed_ptr)
                input_vals.append(val)
            case _ArgMode.PTR:
                # Cast void* to typed pointer, pass pointer directly
                llvm_type = context.get_value_type(spec.numba_type.dtype)
                typed_ptr = builder.bitcast(arg, llvm_type.as_pointer())
                input_vals.append(typed_ptr)
            case _ArgMode.STORE:
                # Cast void* to typed pointer for storing result
                llvm_type = context.get_value_type(spec.numba_type)
                ret_ptr = builder.bitcast(arg, llvm_type.as_pointer())
            case _:
                raise ValueError(f"Invalid arg mode: {spec.mode}")

    # Call the inner function
    cres = context.compile_subroutine(builder, func_device, inner_sig, caching=False)
    result = context.call_internal(builder, cres.fndesc, inner_sig, input_vals)

    # Store result if needed
    if ret_ptr is not None:
        builder.store(result, ret_ptr)

    return context.get_dummy_value()


def _create_void_ptr_wrapper(
    func, name: str, arg_specs: list[_ArgSpec], inner_sig: "Signature"
):
    """
    Given a function and a list of _ArgSpec, create a wrapper function
    that takes all void* arguments, bitcasts them to the
    appropriate typed pointers, and calls the inner function with
    the typed arguments. Each void* argument is handled according
    to its _ArgSpec.

    Args:
        func: The function to wrap (will be compiled as device function)
        name: Base name for the wrapper function
        arg_specs: List of _ArgSpec describing each void* argument
        inner_sig: Numba signature for the inner function call

    Returns:
        Tuple of (wrapper_func, wrapper_sig)
    """
    from numba.cuda import jit as cuda_jit

    # Wrap function as device function
    func_device = cuda_jit(device=True)(func)

    # Generate argument names and signature
    arg_names = [f"arg_{i}" for i in range(len(arg_specs))]
    arg_str = ", ".join(arg_names)
    void_sig = types.void(*(types.voidptr for _ in arg_specs))

    # Create unique wrapper name
    unique_suffix = hex(id(func))[2:]
    wrapper_name = f"wrapped_{name}_{unique_suffix}"

    # We need exec() here because Numba's @intrinsic decorator requires:
    # 1. A function with a specific signature visible at parse time
    # 2. The number of arguments must match the wrapper signature
    # The actual codegen logic is in _codegen_void_ptr_wrapper - this just
    # creates the minimal intrinsic shell that delegates to it.
    wrapper_src = textwrap.dedent(f"""
    @intrinsic
    def impl(typingctx, {arg_str}):
        def codegen(context, builder, impl_sig, args):
            return codegen_helper(context, builder, args, arg_specs, func_device, inner_sig)
        return void_sig, codegen

    def {wrapper_name}({arg_str}):
        return impl({arg_str})
    """)

    local_dict = {
        "intrinsic": intrinsic,
        "void_sig": void_sig,
        "arg_specs": arg_specs,
        "func_device": func_device,
        "inner_sig": inner_sig,
        "codegen_helper": _codegen_void_ptr_wrapper,
    }
    exec(wrapper_src, {}, local_dict)

    wrapper_func = local_dict[wrapper_name]
    wrapper_func.__globals__.update(local_dict)

    return wrapper_func, void_sig


def create_op_void_ptr_wrapper(op, sig: "Signature"):
    """Creates a wrapper function for user-defined operators like unary or binary operators.

    The wrapper takes N+1 arguments where N is the number of input arguments to `op`, the last
    argument is a pointer to the result.
    """
    arg_specs = [_ArgSpec(t, _ArgMode.LOAD) for t in sig.args]
    arg_specs.append(_ArgSpec(sig.return_type, _ArgMode.STORE))
    return _create_void_ptr_wrapper(op, op.__name__, arg_specs, sig)


def create_advance_void_ptr_wrapper(advance_fn, state_ptr_type):
    """Creates a wrapper function for iterator advance method.

    The wrapper takes 2 void* arguments:
    - state pointer
    - offset pointer (points to uint64 value)
    """
    arg_specs = [
        _ArgSpec(state_ptr_type, _ArgMode.PTR),
        _ArgSpec(types.uint64, _ArgMode.LOAD),  # uint64 is the offset type
    ]
    inner_sig = types.void(state_ptr_type, types.uint64)
    return _create_void_ptr_wrapper(
        advance_fn, advance_fn.__name__, arg_specs, inner_sig
    )


def create_input_dereference_void_ptr_wrapper(deref_fn, state_ptr_type, value_type):
    """Creates a wrapper function for input iterator dereference method.

    The wrapper takes 2 void* arguments:
    - state pointer
    - result pointer (function writes result here)
    """
    arg_specs = [
        _ArgSpec(state_ptr_type, _ArgMode.PTR),
        _ArgSpec(types.CPointer(value_type), _ArgMode.PTR),
    ]
    inner_sig = types.void(state_ptr_type, types.CPointer(value_type))
    return _create_void_ptr_wrapper(deref_fn, deref_fn.__name__, arg_specs, inner_sig)


def create_output_dereference_void_ptr_wrapper(deref_fn, state_ptr_type, value_type):
    """Creates a wrapper function for output iterator dereference method.

    The wrapper takes 2 void* arguments:
    - state pointer
    - value pointer (value to write)
    """
    arg_specs = [
        _ArgSpec(state_ptr_type, _ArgMode.PTR),
        _ArgSpec(value_type, _ArgMode.LOAD),
    ]
    inner_sig = types.void(state_ptr_type, value_type)
    return _create_void_ptr_wrapper(deref_fn, deref_fn.__name__, arg_specs, inner_sig)
