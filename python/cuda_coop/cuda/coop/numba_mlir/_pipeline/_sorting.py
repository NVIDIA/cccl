# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Native payload and control contracts shared by sorting families."""

from __future__ import annotations

from numbers import Integral
from typing import Any

from . import _bridges as nb
from . import _templates
from ._errors import TypedPipelineError
from ._types import PRODUCER, CoopGroupType, ThreadDataType


def operations(kinds):
    def make(kind, qualified):
        def operation(group, keys, *args, **kwargs):
            raise NotImplementedError("cuda.coop: kernel only")

        operation.__name__ = f"_coop_{'qualified_' if qualified else ''}{kind}"
        return operation

    return {
        make(kind, qualified): (kind, qualified)
        for kind in sorted(kinds)
        for qualified in (False, True)
    }


def payloads(
    kind, args, qualified, extents=None
) -> tuple[tuple[Any, ...], int | None, bool] | None:
    from .._compiler._parameters import _validate_common_numeric_dtype

    names = ("keys", "values") if kind.endswith("pairs") else ("keys",)
    if len(args) != len(names) + 1 or not isinstance(args[0], CoopGroupType):
        raise TypedPipelineError(f"coop.{kind}: expected group and payloads")
    extents = extents or {}
    dtypes: list[Any] = []
    sizes: list[int | None] = []
    shapes: list[bool] = []
    for name, value in zip(names, args[1:], strict=True):
        array = isinstance(value, nb.types.Array)
        if not qualified and not isinstance(value, ThreadDataType):
            raise TypedPipelineError(
                f"cuda.coop.{kind} requires {name} to be ThreadData"
            )
        if not array:
            raise TypedPipelineError(f"coop.{kind}: {name} must be an array")
        if array and (value.ndim != 1 or value.layout != "C"):
            raise TypedPipelineError(
                f"coop.{kind}: {name} must be contiguous and one-dimensional"
            )
        dtype = value.dtype if array else nb.unliteral(value)
        if not dtype.is_precise():
            return None
        dtypes.append(
            _validate_common_numeric_dtype(
                dtype, operation=kind, parameter=name
            )
        )
        shapes.append(array)
        sizes.append(
            value.items
            if isinstance(value, ThreadDataType)
            else extents.get(name)
            if array
            else 1
        )
    known = {size for size in sizes if size is not None}
    if len(set(shapes)) != 1 or len(known) > 1:
        raise TypedPipelineError(
            f"coop.{kind}: keys and values must have matching shape and extent"
        )
    return tuple(dtypes), sizes[0], shapes[0]


def result_type(value, dtype):
    if isinstance(value, ThreadDataType):
        return ThreadDataType(dtype, value.items, PRODUCER)
    if isinstance(value, nb.types.Array):
        return nb.types.Array(dtype, 1, "C")
    return dtype


def binding(analysis, stmt, name):
    from ..._core import ArgumentBinding
    from ._sites import _RUNTIME

    var = dict(stmt.value.kws).get(name)
    if var is None:
        return ArgumentBinding.omitted()
    scalar = analysis._provenance(var)
    if scalar is _RUNTIME:
        return ArgumentBinding.runtime()
    if scalar.value is None:
        return ArgumentBinding.omitted()
    return ArgumentBinding.static(scalar.value)


def integer_binding(analysis, stmt, kind, name):
    from ..._core import BindingKind
    from .._compiler._parameters import _validate_runtime_integer_dtype

    result = binding(analysis, stmt, name)
    if result.kind is BindingKind.RUNTIME:
        var = dict(stmt.value.kws)[name]
        _validate_runtime_integer_dtype(
            analysis.typemap[var.name], operation=kind, parameter=name
        )
    elif result.kind is BindingKind.STATIC and (
        isinstance(result.value, bool) or not isinstance(result.value, Integral)
    ):
        raise TypedPipelineError(f"coop.{kind}: {name} must be an integer")
    return result


def scalar_operand(name, bound):
    from ..._core import BindingKind

    if bound.kind is BindingKind.STATIC:
        return ("constant", bound.value)
    if bound.kind is not BindingKind.RUNTIME:
        raise AssertionError(f"missing scalar operand {name}")
    return ("value", name)


def register(ops, typing, *, callbacks=()):
    from . import _callbacks
    from ._inference import operation_static_arguments

    for operation, (kind, qualified) in ops.items():
        _templates.OPERATIONS[operation] = kind
        _templates.RESULT_EXTENT_SOURCES[operation] = (
            (1, 2) if kind.endswith("pairs") else 1
        )
        if qualified:
            _templates.QUALIFIED_OPERATIONS.add(operation)
        if callbacks:
            operation_static_arguments[operation] = {
                name: _callbacks.StaticCallableType for name in callbacks
            }
        _templates.register_template(
            operation,
            typing(kind, qualified),
            operation.__name__,
            prefer_literal=True,
        )
    kinds = {kind for kind, _ in ops.values()}
    for public, name in _templates.kernel_primitives().items():
        kind = _templates.group_operation(public)
        if kind not in kinds:
            continue
        qualified = name.startswith("cuda.coop.numba_mlir.")
        operation = next(
            op for op, info in ops.items() if info == (kind, qualified)
        )
        nb.attach_inline_wrapper(
            public, _templates.result_wrapper(public, name, operation)
        )
        _templates.register_template(
            public, _templates._not_inlined(name), name
        )


def site_arguments(analysis, stmt, kind):
    """Read payload extents without taking element typing from Numba."""
    expr = stmt.value
    names = ("keys", "values") if kind.endswith("pairs") else ("keys",)
    extents = {
        name: analysis._value_extent(expr.args[i + 1], kind)
        for i, name in enumerate(names)
    }
    return (
        [analysis.typemap[arg.name] for arg in expr.args],
        {name: analysis.typemap[var.name] for name, var in expr.kws},
        extents,
    )
