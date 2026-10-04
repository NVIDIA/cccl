# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Infer reduction provider inputs and check valid-prefix controls.

Both scalar values and fixed per-thread arrays reduce to one scalar result.
These hooks recover the input dtype and form for factory specialization, then
check any CUB valid-prefix count before provider materialization. Runtime
count conversion is handled by the provider's bounded integer ABI.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any

from cuda.coop._core import ArgumentBinding, BindingKind

from ._group_rewriting import GroupRewriteContext
from ._parameters import (
    _validate_common_numeric_dtype,
    _validate_runtime_integer_dtype,
)
from ._rewrite_payload import PayloadInference
from ._rewrite_support import CoopSinglePhaseRewriteError, ir


def _numeric_dtype(dtype: Any, *, binary_op: Any) -> Any:
    """Validate numeric payload types against a built-in or custom operator.

    Recognized built-ins apply their extra dtype rules, such as integer-only
    bitwise reduction. A custom callable uses the common numeric set; later
    callback compilation checks its concrete signature. Translate validation
    failures to the provider rewrite's error type.
    """

    from .._lowering._reduce import (
        normalize_reduce_operation,
        validate_reduce_operator_dtype,
    )

    try:
        operation = normalize_reduce_operation(binary_op)
    except NotImplementedError:
        operation = None
    try:
        if operation is None:
            return _validate_common_numeric_dtype(
                dtype,
                operation="reduce",
                parameter="value",
            )
        return validate_reduce_operator_dtype(operation, dtype)
    except (TypeError, ValueError) as exc:
        raise CoopSinglePhaseRewriteError(str(exc)) from exc


def infer_reduce_payload(
    context: GroupRewriteContext,
    inference: PayloadInference,
) -> None:
    """Infer the reduction value's dtype, extent, and scalar/array form.

    A recognized array must have a static extent. Otherwise infer a scalar
    with one item. Use payload or scalar type facts before an explicit factory
    dtype, then apply the operator's numeric restrictions.

    Merge the resulting factory keywords through ``inference`` so explicit
    values must agree. For arrays, record dtype on the payload for later
    allocation. The reduction result remains scalar in either input form.
    """

    if not inference.runtime_args or not isinstance(
        inference.runtime_args[0], ir.Var
    ):
        raise CoopSinglePhaseRewriteError(
            "coop reduce value must be a runtime scalar or fixed-size array"
        )
    value = inference.runtime_args[0]
    array_var, array_specification = inference.array_candidate(0)
    if array_specification is not None:
        items_per_thread = array_specification.items_per_thread
        if items_per_thread is None:
            raise CoopSinglePhaseRewriteError(
                "coop reduce array value must have "
                "a static items_per_thread extent"
            )
        dtype = inference.inferred_array_dtype(array_var, array_specification)
    else:
        items_per_thread = 1
        dtype = context.dtype(value)
    if dtype is None:
        dtype = inference.factory_value("dtype")
    if dtype is None:
        raise CoopSinglePhaseRewriteError(
            "coop reduce could not infer value dtype"
        )

    dtype = _numeric_dtype(
        dtype,
        binary_op=inference.factory_value("binary_op"),
    )
    inference.infer_kwarg("dtype", dtype)
    inference.infer_kwarg("items_per_thread", items_per_thread)
    inference.infer_kwarg(
        "value_kind",
        "array" if array_specification is not None else "scalar",
    )
    if array_specification is not None:
        context.record_thread_data_dtype(value, dtype)


def _group_width(
    factory_kwargs: dict[str, object], *, parameter: str
) -> int | None:
    """Recover the member count used to bound a valid-prefix control.

    Warp providers use ``threads_in_warp``. Block providers use a scalar block
    size or the product of a three-dimensional shape. Return ``None`` when
    these factory inputs do not establish an integer width.
    """

    if parameter == "valid_items":
        value = factory_kwargs.get("threads_in_warp")
        if isinstance(value, Integral) and not isinstance(value, bool):
            return int(value)
        return None
    value = factory_kwargs.get("threads_per_block")
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    if (
        isinstance(value, tuple)
        and len(value) == 3
        and all(
            isinstance(component, Integral) and not isinstance(component, bool)
            for component in value
        )
    ):
        return int(value[0]) * int(value[1]) * int(value[2])
    return None


def _validate_valid_items(
    context: GroupRewriteContext,
    *,
    runtime_args: list[ir.Var],
    factory_kwargs: dict[str, object],
    parameter: str,
) -> None:
    """Check the representation and known bounds of a CUB valid prefix.

    An absent control needs no check. A static binding must be a positive
    non-boolean integer and cannot exceed a known group width. A runtime
    binding requires a second operand and a supported integer dtype when that
    type is known. Unknown runtime type defers to later typing.

    This hook validates call structure and known facts. The factory supplies
    the bounded conversion for runtime values before the native call.
    """

    binding = factory_kwargs.get(parameter)
    if binding is None:
        return
    if not isinstance(binding, ArgumentBinding):
        raise CoopSinglePhaseRewriteError(
            f"coop reduce {parameter} must be an integer binding"
        )
    group_width = _group_width(factory_kwargs, parameter=parameter)
    if binding.kind is BindingKind.STATIC:
        value = binding.value
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise CoopSinglePhaseRewriteError(
                f"coop reduce {parameter} must be an integer, not bool"
            )
        value = int(value)
        if value < 1:
            raise CoopSinglePhaseRewriteError(
                f"coop reduce {parameter} must be at least 1"
            )
        if group_width is not None and value > group_width:
            raise CoopSinglePhaseRewriteError(
                f"coop reduce {parameter} {value} "
                f"exceeds group size {group_width}"
            )
        return
    if binding.kind is not BindingKind.RUNTIME:
        raise CoopSinglePhaseRewriteError(
            f"coop reduce {parameter} must be an integer binding"
        )
    if len(runtime_args) != 2 or not isinstance(runtime_args[1], ir.Var):
        raise CoopSinglePhaseRewriteError(
            f"coop reduce runtime {parameter} is missing its runtime value"
        )
    dtype = context.numba_type(runtime_args[1])
    if dtype is None:
        dtype = context.dtype(runtime_args[1])
    if dtype is None:
        return
    try:
        _validate_runtime_integer_dtype(
            dtype,
            operation="reduce",
            parameter="valid_items",
        )
    except TypeError as exc:
        raise CoopSinglePhaseRewriteError(str(exc)) from exc


def validate_block_reduce_runtime_controls(
    context: GroupRewriteContext,
    *,
    op_name: str,
    runtime_args: list[ir.Var],
    factory_kwargs: dict[str, object],
) -> None:
    """Validate direct BlockReduce valid-prefix controls."""

    del op_name
    _validate_valid_items(
        context,
        runtime_args=runtime_args,
        factory_kwargs=factory_kwargs,
        parameter="num_valid",
    )


def validate_warp_reduce_runtime_controls(
    context: GroupRewriteContext,
    *,
    op_name: str,
    runtime_args: list[ir.Var],
    factory_kwargs: dict[str, object],
) -> None:
    """Validate direct WarpReduce valid-prefix controls."""

    del op_name
    _validate_valid_items(
        context,
        runtime_args=runtime_args,
        factory_kwargs=factory_kwargs,
        parameter="valid_items",
    )


__all__ = [
    "infer_reduce_payload",
    "validate_block_reduce_runtime_controls",
    "validate_warp_reduce_runtime_controls",
]
