# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numba_cuda_mlir import types

from ._numba_mlir_compat import _get_numba_mlir_compat

ir = _get_numba_mlir_compat().numba_ir


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
    value: Any,
    *,
    definitions: Callable[[ir.Var], Iterable[Any]],
    argument_type: Callable[[int], Any | None],
    seen: set[str] | None = None,
) -> tuple[bool, StaticScalarProvenance | None]:
    """Resolve a scalar and retain whether Numba already assigned its dtype."""

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
            if isinstance(arg_type, types.Literal):
                resolved_values.append(
                    _static_scalar(
                        arg_type.literal_value, arg_type.literal_type
                    )
                )
                continue
            if isinstance(arg_type, types.NoneType) or (
                isinstance(arg_type, types.Omitted) and arg_type.value is None
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
    value: Any,
    *,
    definitions: Callable[[ir.Var], Iterable[Any]],
    argument_type: Callable[[int], Any | None],
    seen: set[str] | None = None,
) -> tuple[bool, Any]:
    """Resolve a scalar only when every reaching definition is static.

    Globals, free variables, literals, and IR constants are static. Aliases,
    casts, and phi nodes preserve that classification only when all incoming
    definitions resolve to the same typed value. Runtime expressions are never
    evaluated through Numba's general constant-inference machinery here.
    """

    resolved, scalar = try_resolve_static_scalar_provenance(
        value,
        definitions=definitions,
        argument_type=argument_type,
        seen=seen,
    )
    return (resolved, None if scalar is None else _typed_static_value(scalar))


def scalar_expression_dtype(definition, dtype):
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


def scalar_call_dtype(function, arguments, dtype):
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


def cuda_index_dtype(definition, attribute_chain, cuda_module):
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
            return types.int32
    return None


__all__: tuple[str, ...] = ()
