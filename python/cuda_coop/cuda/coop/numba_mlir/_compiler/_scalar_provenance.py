# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared scalar provenance helpers for cooperative IR planning."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from numba_cuda_mlir import types
from numba_cuda_mlir.numbair_transforms import ir


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

    if not isinstance(value, ir.Var):
        return (True, value)
    if seen is None:
        seen = set()
    if value.name in seen:
        return (False, None)
    seen.add(value.name)

    resolved_values: list[Any] = []
    for definition in definitions(value):
        if isinstance(definition, ir.Arg):
            arg_type = argument_type(definition.index)
            if isinstance(arg_type, types.Literal):
                resolved_values.append(arg_type.literal_value)
                continue
            if isinstance(arg_type, types.NoneType) or (
                isinstance(arg_type, types.Omitted) and arg_type.value is None
            ):
                resolved_values.append(None)
                continue
            return (False, None)
        if isinstance(definition, (ir.Global, ir.FreeVar, ir.Const)):
            resolved_values.append(definition.value)
            continue
        if isinstance(definition, ir.Var):
            resolved, scalar = try_resolve_static_scalar(
                definition,
                definitions=definitions,
                argument_type=argument_type,
                seen=set(seen),
            )
            if not resolved:
                return (False, None)
            resolved_values.append(scalar)
            continue
        if isinstance(definition, ir.Expr) and definition.op == "cast":
            resolved, scalar = try_resolve_static_scalar(
                definition.value,
                definitions=definitions,
                argument_type=argument_type,
                seen=set(seen),
            )
            if not resolved:
                return (False, None)
            resolved_values.append(scalar)
            continue
        if isinstance(definition, ir.Expr) and definition.op == "phi":
            incoming_values = getattr(definition, "incoming_values", ())
            if not isinstance(incoming_values, (list, tuple)) or not incoming_values:
                return (False, None)
            for incoming in incoming_values:
                resolved, scalar = try_resolve_static_scalar(
                    incoming,
                    definitions=definitions,
                    argument_type=argument_type,
                    seen=set(seen),
                )
                if not resolved:
                    return (False, None)
                resolved_values.append(scalar)
            continue
        return (False, None)

    if not resolved_values:
        return (False, None)
    first = resolved_values[0]
    if any(
        type(candidate) is not type(first) or candidate != first
        for candidate in resolved_values[1:]
    ):
        return (False, None)
    return (True, first)


__all__: tuple[str, ...] = ()
