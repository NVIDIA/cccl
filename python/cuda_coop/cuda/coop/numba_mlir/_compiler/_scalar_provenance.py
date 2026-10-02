# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Track compile-time scalar values without losing their original numeric type.

A Python literal may take the element type required by a cooperative operation,
but a value explicitly typed as ``int64`` must not silently become ``int32``.
``StaticScalarProvenance`` keeps the value together with any dtype already
attached to it. Here, "provenance" means the constants, arguments, and
forwarding assignments that establish both facts.

Group planning and call rewriting share this analysis when deciding whether a
control such as ``offset`` or ``oob_default`` can be embedded in generated code.
The static-value resolver accepts only explicit constants and literal arguments
whose possible definitions agree; it does not evaluate runtime expressions to
make them static. Separate helpers infer result types for scalar operators,
casts, and CUDA indices so runtime values can still be checked against an
operation's numeric requirements. Neither analysis rewrites the function IR or
requests a new compiler specialization.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Any

import numba_cuda_mlir.numba_cuda.types as _numba_types
import numpy as np

if TYPE_CHECKING:
    from numba_cuda_mlir.numba_cuda.core import ir
else:
    from numba_cuda_mlir.numbair_transforms import ir


@dataclass(frozen=True)
class StaticScalarProvenance:
    """A static scalar and any compiler dtype already attached to it."""

    value: Any
    dtype: Any | None = None


def _static_scalar(
    value: Any, dtype: Any | None = None
) -> StaticScalarProvenance:
    if dtype is None and isinstance(value, np.generic):
        dtype = value.dtype
    return StaticScalarProvenance(value=value, dtype=dtype)


def _typed_static_value(scalar: StaticScalarProvenance) -> Any:
    """Keep a compiler literal's scalar dtype after unwrapping provenance."""

    if scalar.dtype is None or isinstance(scalar.value, np.generic):
        return scalar.value
    try:
        return np.dtype(str(scalar.dtype)).type(scalar.value)
    except (TypeError, ValueError, OverflowError):
        return scalar.value


def try_resolve_static_scalar_provenance(
    value: object,
    *,
    definitions: Callable[[ir.Var], Iterable[object]],
    argument_type: Callable[[int], _numba_types.Type | None],
    seen: set[str] | None = None,
) -> tuple[bool, StaticScalarProvenance | None]:
    """Resolve an explicitly static value while retaining its compiler dtype.

    Planning must distinguish a literal supplied by the user from a runtime
    expression that general constant inference happens to evaluate. Accept
    constants, globals, free variables, literal arguments, and arguments known
    to be ``None``; follow only aliases, casts, and phi inputs. Every reaching
    leaf must agree in Python value type, value, and recorded dtype. Runtime
    expressions, unresolved paths, and cycles make the result unresolved. This
    traversal neither evaluates scalar operators nor requests dispatcher
    specialization.

    Parameters
    ----------
    value : object
        Value to inspect. Non-variable inputs are accepted directly as static;
        callers are responsible for restricting them to the intended scalar
        domain.
    definitions : callable
        Return all reaching definitions for an IR variable.
    argument_type : callable
        Return the compiler type for a function argument index, or ``None`` when
        unavailable. Literal types retain their ``literal_type``.
    seen : set of str, optional
        Names already visited on this recursion path. The current variable is
        added in place; recursive branches receive separate copies.

    Returns
    -------
    resolved : bool
        Whether all inspected definitions establish the same static value.
    scalar : StaticScalarProvenance or None
        Resolved value and its known dtype, or ``None`` on failure. A resolved
        ``None`` value is represented by a provenance object and is distinct
        from failure. NumPy scalars contribute their own dtype; ordinary Python
        constants leave the dtype unspecified for contextual coercion.
    """

    if not isinstance(value, ir.Var):
        return (True, _static_scalar(value))
    if seen is None:
        seen = set()
    if value.name in seen:
        return (False, None)
    seen.add(value.name)

    resolved_values: list[StaticScalarProvenance] = []
    for definition in definitions(value):
        if isinstance(definition, ir.Arg):
            arg_type = argument_type(definition.index)
            if isinstance(arg_type, _numba_types.Literal):
                resolved_values.append(
                    _static_scalar(
                        arg_type.literal_value, arg_type.literal_type
                    )
                )
                continue
            if isinstance(arg_type, _numba_types.NoneType) or (
                isinstance(arg_type, _numba_types.Omitted)
                and arg_type.value is None
            ):
                resolved_values.append(_static_scalar(None, arg_type))
                continue
            return (False, None)
        if isinstance(definition, (ir.Global, ir.FreeVar, ir.Const)):
            resolved_values.append(_static_scalar(definition.value))
            continue
        if isinstance(definition, ir.Var):
            resolved, scalar = try_resolve_static_scalar_provenance(
                definition,
                definitions=definitions,
                argument_type=argument_type,
                seen=set(seen),
            )
            if not resolved or scalar is None:
                return (False, None)
            resolved_values.append(scalar)
            continue
        if isinstance(definition, ir.Expr) and definition.op == "cast":
            resolved, scalar = try_resolve_static_scalar_provenance(
                definition.value,
                definitions=definitions,
                argument_type=argument_type,
                seen=set(seen),
            )
            if not resolved or scalar is None:
                return (False, None)
            resolved_values.append(scalar)
            continue
        if isinstance(definition, ir.Expr) and definition.op == "phi":
            incoming_values = getattr(definition, "incoming_values", ())
            if (
                not isinstance(incoming_values, (list, tuple))
                or not incoming_values
            ):
                return (False, None)
            for incoming in incoming_values:
                resolved, scalar = try_resolve_static_scalar_provenance(
                    incoming,
                    definitions=definitions,
                    argument_type=argument_type,
                    seen=set(seen),
                )
                if not resolved or scalar is None:
                    return (False, None)
                resolved_values.append(scalar)
            continue
        return (False, None)

    if not resolved_values:
        return (False, None)
    first = resolved_values[0]
    if any(
        type(candidate.value) is not type(first.value)
        or candidate.value != first.value
        or candidate.dtype != first.dtype
        for candidate in resolved_values[1:]
    ):
        return (False, None)
    return (True, first)


def try_resolve_static_scalar(
    value: object,
    *,
    definitions: Callable[[ir.Var], Iterable[object]],
    argument_type: Callable[[int], _numba_types.Type | None],
    seen: set[str] | None = None,
) -> tuple[bool, Any]:
    """Resolve a static value and preserve a known scalar width when possible.

    Use ``try_resolve_static_scalar_provenance`` to require agreement across all
    reaching definitions without evaluating runtime expressions. Unwrap its
    result, converting a value with a known compiler dtype to the matching NumPy
    scalar when possible. If that conversion is unsupported or fails, return the
    original value. Use the provenance-returning helper when the recorded dtype
    itself is needed for validation.

    Parameters
    ----------
    value : object
        IR value or already-static Python value to resolve.
    definitions : callable
        Return all reaching definitions for an IR variable.
    argument_type : callable
        Return the compiler type for a function argument index, or ``None`` when
        unavailable.
    seen : set of str, optional
        Names on the current recursion path. Passed to the provenance resolver,
        which adds the current variable in place and copies it for branches.

    Returns
    -------
    resolved : bool
        Whether the value has consistent, explicitly static provenance.
    value : object
        Unwrapped static value, or ``None`` on failure. Consult ``resolved`` to
        distinguish an unresolved value from a statically known ``None``.
    """

    resolved, scalar = try_resolve_static_scalar_provenance(
        value,
        definitions=definitions,
        argument_type=argument_type,
        seen=seen,
    )
    return (resolved, None if scalar is None else _typed_static_value(scalar))


def scalar_expression_dtype(
    definition: ir.Expr,
    dtype: Callable[[ir.Var], object],
) -> _numba_types.Type | None:
    """Infer a scalar operator's result from the operand types already known."""

    from ._parameters import _scalar_operator_result_dtype

    if definition.op in {"binop", "inplace_binop"}:
        operands = (definition.lhs, definition.rhs)
    else:
        operands = (definition.value,)
    return _scalar_operator_result_dtype(
        definition.fn,
        *(
            dtype(value) if isinstance(value, ir.Var) else None
            for value in operands
        ),
    )


def scalar_call_dtype(
    function: object,
    arguments: Sequence[object],
    dtype: Callable[[ir.Var], object],
) -> _numba_types.Type | None:
    """Infer an explicit scalar cast, retaining the compiler's numeric rules."""

    from ._parameters import _scalar_cast_dtype, _scalar_operator_result_dtype

    cast_dtype = _scalar_cast_dtype(function)
    if cast_dtype is None:
        return None
    if len(arguments) == 1 and isinstance(arguments[0], ir.Var):
        inferred = _scalar_operator_result_dtype(function, dtype(arguments[0]))
        if inferred is not None:
            return inferred
    return cast_dtype


def cuda_index_dtype(
    definition: ir.Expr,
    attribute_chain: Callable[[ir.Var], tuple[object, Sequence[str]] | None],
    cuda_module: ModuleType,
) -> _numba_types.Type | None:
    """Return the compiler dtype of CUDA launch indices and dimensions."""

    if definition.op != "getattr":
        return None
    chain = attribute_chain(definition.value)
    if chain is not None:
        root, attributes = chain
        if root is cuda_module and (*attributes, definition.attr) in {
            (index, component)
            for index in ("blockDim", "blockIdx", "gridDim", "threadIdx")
            for component in ("x", "y", "z")
        }:
            return _numba_types.int32
    return None


__all__: tuple[str, ...] = ()
