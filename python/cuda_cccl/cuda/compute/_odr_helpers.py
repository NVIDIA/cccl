# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
ODR (One Definition Rule) Helpers for CCCL Python Interop.

This module provides utilities to create wrapper functions for
device functions that are defined in Python and JIT compiled by numba-cuda-mlir.

On the C++ side, these functions are declared as `extern "C"` functions with
void* parameters - the argument types can not be known at C++ compile time.

Thus, the helpers in this module generate wrapper device functions that accept
void* arguments (matching C++ declarations), reinterpret them as the correct
typed pointers, load/store values as needed, and call the original function
with properly typed arguments.

Example flow:
    User provides: def add(x: int32, y: int32) -> int32
    Wrapper signature: void(void*, void*, void*)  # x_ptr, y_ptr, result_ptr
    C++ sees: extern "C" void wrapped_add(void*, void*, void*);

Unlike the previous numba-cuda implementation, the wrappers here are *ordinary
Python device functions* compiled with ``abi="c"`` rather than hand-written
LLVM-IR codegen (``@intrinsic``).  A ``void*`` argument is expressed as a typed
``CPointer`` parameter (ABI-identical to ``void*``); loads/stores become
``ptr[0]`` indexing.  numba-cuda-mlir inlines the user operator into the
wrapper, so the generated code is equivalent to the old codegen without any
low-level builder work.
"""

from __future__ import annotations

import itertools
import threading

from ._mlir import cuda, types
from ._utils import sanitize_identifier

# Global counter to generate unique symbol names even when the same function
# is used multiple times (e.g., as both selectors in `three_way_partition`).
_wrapper_name_counter = itertools.count()
_wrapper_name_lock = threading.Lock()

__all__ = [
    "create_op_void_ptr_wrapper",
    "create_stateful_op_void_ptr_wrapper",
]


def _make_wrapper_name(name: str) -> str:
    """Build a unique, valid C identifier for a generated wrapper."""
    sanitized_name = sanitize_identifier(name)
    if not sanitized_name.isidentifier():
        raise ValueError(
            f"Function name '{name}' cannot be sanitized into a valid identifier"
        )
    with _wrapper_name_lock:
        unique_suffix = next(_wrapper_name_counter)
    return f"wrapped_{sanitized_name}_{unique_suffix}"


def _build_wrapper(
    wrapper_name: str, params: list[str], body_stmts, op_device, extra_namespace=None
):
    """exec a generated wrapper source and return the resulting function.

    ``params`` are the wrapper's parameter names and ``body_stmts`` is a list of
    (unindented) statement lines for its body.  ``op_device`` is injected as
    ``_op`` so the body can call the compiled user operator; ``extra_namespace``
    injects any other globals the body references.
    """
    indented_body = "\n".join(f"    {stmt}" for stmt in body_stmts)
    src = f"def {wrapper_name}({', '.join(params)}):\n{indented_body}\n"
    namespace: dict = {"_op": op_device}
    if extra_namespace:
        namespace.update(extra_namespace)
    exec(src, namespace)
    return namespace[wrapper_name]


def _is_gpu_struct_type(numba_type):
    """True if ``numba_type`` is a registered gpu_struct type (see _jit)."""
    return hasattr(numba_type, "_field_spec") and hasattr(numba_type, "python_type")


def _op_returns_tuple(op_device, arg_types) -> bool:
    """Whether ``op`` naturally returns a tuple for the given argument types."""
    _, op_return_type = cuda.compile(
        op_device, tuple(arg_types), device=True, output="ltoir"
    )
    return isinstance(op_return_type, (types.Tuple, types.UniTuple))


def _result_store_body(loads: str, return_type, reconstruct_from_tuple: bool):
    """Build the wrapper body that computes the op result and stores it.

    A struct-returning operator usually returns the struct directly, which is
    stored as-is.  But an operator can also return a *tuple* of the struct's
    field values (e.g. a scan op feeding a zip output iterator returns a tuple);
    numba-cuda-mlir cannot store a tuple directly into a struct pointer, so when
    the op returns a tuple we reconstruct the struct field-by-field and let the
    gpu_struct constructor pack it.  Returns ``(body_stmts, extra_namespace)``.
    """
    if reconstruct_from_tuple and _is_gpu_struct_type(return_type):
        num_fields = len(return_type._field_spec)
        fields = ", ".join(f"_r[{i}]" for i in range(num_fields))
        stmts = [f"_r = _op({loads})", f"result[0] = _ResultStruct({fields})"]
        return stmts, {"_ResultStruct": return_type.python_type}
    return [f"result[0] = _op({loads})"], {}


def create_op_void_ptr_wrapper(op, sig):
    """Create a wrapper for a stateless user operator (unary, binary, ...).

    The wrapper takes ``N + 1`` ``void*`` arguments where ``N`` is the number of
    inputs to ``op``; the trailing argument is a pointer to the result storage.

    Returns ``(wrapper_func, wrapper_sig)``.
    """
    op_device = cuda.jit(device=True)(op)

    arg_types = list(sig.args)
    return_type = sig.return_type

    wrapper_name = _make_wrapper_name(op.__name__)
    arg_names = [f"arg_{i}" for i in range(len(arg_types))]

    # result[0] = _op(arg_0[0], arg_1[0], ...)
    loads = ", ".join(f"{name}[0]" for name in arg_names)
    reconstruct = _is_gpu_struct_type(return_type) and _op_returns_tuple(
        op_device, arg_types
    )
    body, extra_namespace = _result_store_body(loads, return_type, reconstruct)

    wrapper_func = _build_wrapper(
        wrapper_name, arg_names + ["result"], body, op_device, extra_namespace
    )

    wrapper_sig = types.void(
        *(types.CPointer(t) for t in arg_types),
        types.CPointer(return_type),
    )
    return wrapper_func, wrapper_sig


def create_stateful_op_void_ptr_wrapper(op, sig, state_dtypes):
    """Create a wrapper for a stateful operator.

    A stateful operator captures one or more device arrays as state.  The
    transformed ``op`` takes those state arrays first, followed by the regular
    inputs (see ``_jit._compile_stateful_op``).  On the C++ side the state is a
    single ``void*`` pointing to a packed array of the state data pointers.

    The wrapper takes ``2 + K`` ``void*`` arguments:
    - ``states``: pointer to the packed array of state data pointers,
    - ``K`` regular inputs (one per non-state argument of ``op``),
    - ``result``: pointer to the result storage.

    ``state_dtypes`` is the list of numba-cuda-mlir scalar types of the state
    arrays.  All state arrays must share a dtype: the packed pointers are read
    through a single ``CPointer(CPointer(dtype))`` view, which requires a
    uniform pointee type.  Heterogeneous state dtypes are not yet supported
    (reinterpreting raw addresses to differently-typed pointers has no
    pure-Python expression in numba-cuda-mlir).

    Returns ``(wrapper_func, wrapper_sig)``.
    """
    num_states = len(state_dtypes)
    if num_states == 0:
        raise ValueError("stateful op wrapper requires at least one state array")

    unique_state_dtypes = set(state_dtypes)
    if len(unique_state_dtypes) > 1:
        raise NotImplementedError(
            "stateful operators that capture device arrays of differing dtypes "
            f"are not supported (got {sorted(map(str, unique_state_dtypes))}); "
            "all captured arrays must share a dtype"
        )
    state_dtype = state_dtypes[0]

    op_device = cuda.jit(device=True)(op)

    # sig.args == (state_0, ..., state_{num_states-1}, input_0, ..., input_{K-1})
    input_types = list(sig.args)[num_states:]
    return_type = sig.return_type

    wrapper_name = _make_wrapper_name(op.__name__)
    input_names = [f"arg_{i}" for i in range(len(input_types))]

    # states[j] reinterprets the j-th packed pointer as CPointer(state_dtype).
    state_args = ", ".join(f"states[{j}]" for j in range(num_states))
    input_args = ", ".join(f"{name}[0]" for name in input_names)
    call_args = ", ".join(a for a in (state_args, input_args) if a)
    reconstruct = _is_gpu_struct_type(return_type) and _op_returns_tuple(
        op_device, sig.args
    )
    body, extra_namespace = _result_store_body(call_args, return_type, reconstruct)

    wrapper_func = _build_wrapper(
        wrapper_name,
        ["states", *input_names, "result"],
        body,
        op_device,
        extra_namespace,
    )

    wrapper_sig = types.void(
        types.CPointer(types.CPointer(state_dtype)),
        *(types.CPointer(t) for t in input_types),
        types.CPointer(return_type),
    )
    return wrapper_func, wrapper_sig
