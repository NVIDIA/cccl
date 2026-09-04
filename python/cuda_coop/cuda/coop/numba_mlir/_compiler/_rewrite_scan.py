# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Scan payload inference and scalar-control validation."""

from __future__ import annotations

from numbers import Integral
from typing import Any

from numba_cuda_mlir.numbair_transforms import ir

from cuda.coop._core import ArgumentBinding, BindingKind

from ._group_rewriting import GroupRewriteContext
from ._parameters import (
    _validate_common_numeric_dtype,
    _validate_runtime_integer_dtype,
    coerce_static_scalar,
)
from ._rewrite_payload import PayloadInference
from ._rewrite_support import (
    _UNRESOLVED,
    CoopSinglePhaseRewriteError,
    _dtype_values_match,
)


def _runtime_binding(value: object) -> bool:
    return isinstance(value, ArgumentBinding) and value.kind is BindingKind.RUNTIME


def _payload_dtype(
    context: GroupRewriteContext,
    value: ir.Var | None,
    spec: Any,
) -> Any | None:
    dtype = spec.dtype if spec is not None else None
    if dtype is None and value is not None:
        dtype = context.dtype(value)
    if dtype is None and value is not None and spec is not None:
        dtype = context.infer_thread_data_write_dtype(value)
    return dtype


def _validate_aggregate(
    context: GroupRewriteContext,
    inference: PayloadInference,
    *,
    index: int,
    dtype: Any,
) -> None:
    aggregate, spec = inference.array_candidate(index)
    if aggregate is None or spec is None or spec.items_per_thread != 1:
        raise CoopSinglePhaseRewriteError(
            "coop scan aggregate_output must be a one-item ThreadData or local array"
        )
    aggregate_dtype = _payload_dtype(context, aggregate, spec)
    if aggregate_dtype is not None and not _dtype_values_match(
        aggregate_dtype,
        dtype,
    ):
        raise CoopSinglePhaseRewriteError(
            "coop scan aggregate_output dtype must exactly match the value dtype"
        )
    context.record_thread_data_dtype(aggregate, dtype)


def _runtime_initial_index(
    *,
    base_count: int,
    factory_kwargs: dict[str, object],
) -> int | None:
    if _runtime_binding(factory_kwargs.get("initial_value")):
        return base_count
    return None


def _validate_initial_value(
    context: GroupRewriteContext,
    *,
    runtime_args: list[ir.Var],
    factory_kwargs: dict[str, object],
    dtype: Any,
    base_count: int,
) -> None:
    initial = factory_kwargs.get("initial_value")
    if not isinstance(initial, ArgumentBinding):
        return
    if initial.kind is BindingKind.STATIC:
        provenance = context.static_scalar_provenance(initial.value)
        source_dtype = (
            None
            if provenance is _UNRESOLVED or provenance is None
            else provenance.dtype
        )
        try:
            value = coerce_static_scalar(
                initial.value,
                dtype,
                operation="scan",
                parameter="initial_value",
                source_dtype=source_dtype,
            )
        except (TypeError, ValueError) as exc:
            raise CoopSinglePhaseRewriteError(str(exc)) from exc
        factory_kwargs["initial_value"] = ArgumentBinding.static(value)
        return
    if initial.kind is not BindingKind.RUNTIME:
        return
    index = _runtime_initial_index(
        base_count=base_count,
        factory_kwargs=factory_kwargs,
    )
    if index is None or index >= len(runtime_args):
        raise CoopSinglePhaseRewriteError(
            "coop scan runtime initial_value is missing its runtime value"
        )
    value = runtime_args[index]
    initial_dtype = context.numba_type(value)
    if initial_dtype is None:
        initial_dtype = context.dtype(value)
    if initial_dtype is None:
        return
    try:
        initial_dtype = _validate_common_numeric_dtype(
            initial_dtype,
            operation="scan",
            parameter="initial_value",
        )
    except (TypeError, ValueError) as exc:
        raise CoopSinglePhaseRewriteError(str(exc)) from exc
    if not _dtype_values_match(initial_dtype, dtype):
        raise CoopSinglePhaseRewriteError(
            "coop scan runtime initial_value dtype must exactly match the "
            f"value dtype {dtype}; got {initial_dtype}"
        )


def infer_scan_payload(
    context: GroupRewriteContext,
    inference: PayloadInference,
) -> None:
    """Infer Scan payload and side-output dtype metadata."""

    is_block_array = inference.op_name == "block_scan_array"
    if is_block_array:
        input_value, input_spec = inference.array_candidate(0)
        output_value, output_spec = inference.array_candidate(1)
        if input_spec is None or output_spec is None:
            raise CoopSinglePhaseRewriteError(
                "coop block scan array providers require input and output arrays"
            )
        if (
            input_spec.items_per_thread is None
            or input_spec.items_per_thread != output_spec.items_per_thread
        ):
            raise CoopSinglePhaseRewriteError(
                "coop block scan input and output arrays must have the same "
                "static items_per_thread extent"
            )
        input_dtype = _payload_dtype(context, input_value, input_spec)
        output_dtype = _payload_dtype(context, output_value, output_spec)
        if (
            input_dtype is not None
            and output_dtype is not None
            and not (_dtype_values_match(input_dtype, output_dtype))
        ):
            raise CoopSinglePhaseRewriteError(
                "coop block scan input and output arrays must have exactly "
                "matching dtypes"
            )
        dtype = input_dtype if input_dtype is not None else output_dtype
        if dtype is None:
            dtype = inference.factory_value("dtype")
        inference.infer_kwarg("items_per_thread", input_spec.items_per_thread)
        inference.infer_kwarg("value_kind", "array")
        if dtype is not None:
            for value in (input_value, output_value):
                if value is not None:
                    context.record_thread_data_dtype(value, dtype)
        base_count = 2
    else:
        if not inference.runtime_args or not isinstance(
            inference.runtime_args[0], ir.Var
        ):
            raise CoopSinglePhaseRewriteError(
                "coop scan value must be a runtime scalar"
            )
        value = inference.runtime_args[0]
        if context.array(value) is not None:
            raise CoopSinglePhaseRewriteError(
                "coop scalar scan providers do not accept array values"
            )
        dtype = context.dtype(value)
        if dtype is None:
            dtype = inference.factory_value("dtype")
        inference.infer_kwarg("items_per_thread", 1)
        inference.infer_kwarg("value_kind", "scalar")
        base_count = 1

    if dtype is None:
        raise CoopSinglePhaseRewriteError("coop scan could not infer value dtype")
    try:
        dtype = _validate_common_numeric_dtype(
            dtype,
            operation="scan",
            parameter="value",
        )
        from .._lowering._scan import validate_scan_operator_dtype

        validate_scan_operator_dtype(
            inference.factory_value("scan_op"),
            dtype,
        )
    except (TypeError, ValueError) as exc:
        raise CoopSinglePhaseRewriteError(str(exc)) from exc
    inference.infer_kwarg("dtype", dtype)
    _validate_initial_value(
        context,
        runtime_args=inference.runtime_args,
        factory_kwargs=inference.factory_kwargs,
        dtype=dtype,
        base_count=base_count,
    )

    cursor = base_count
    if _runtime_binding(inference.factory_kwargs.get("initial_value")):
        cursor += 1
    if _runtime_binding(inference.factory_kwargs.get("valid_items")):
        cursor += 1
    aggregate_name = (
        "block_aggregate"
        if inference.op_name.startswith("block_scan_")
        else "warp_aggregate"
    )
    if inference.factory_kwargs.get(aggregate_name):
        _validate_aggregate(
            context,
            inference,
            index=cursor,
            dtype=dtype,
        )


def validate_scan_runtime_controls(
    context: GroupRewriteContext,
    *,
    op_name: str,
    runtime_args: list[ir.Var],
    factory_kwargs: dict[str, object],
) -> None:
    """Validate runtime and static WarpScan prefix-width controls."""

    del op_name
    valid_items = factory_kwargs.get("valid_items")
    if not isinstance(valid_items, ArgumentBinding):
        return
    width = factory_kwargs.get("threads_in_warp")
    if valid_items.kind is BindingKind.STATIC:
        value = valid_items.value
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise CoopSinglePhaseRewriteError(
                "coop scan valid_items must be an integer, not bool"
            )
        value = int(value)
        if value < 1:
            raise CoopSinglePhaseRewriteError(
                "coop scan valid_items must be at least 1"
            )
        if isinstance(width, Integral) and value > int(width):
            raise CoopSinglePhaseRewriteError(
                f"coop scan valid_items {value} exceeds group size {width}"
            )
        return
    if valid_items.kind is not BindingKind.RUNTIME:
        return
    cursor = 1
    if _runtime_binding(factory_kwargs.get("initial_value")):
        cursor += 1
    if cursor >= len(runtime_args):
        raise CoopSinglePhaseRewriteError(
            "coop scan runtime valid_items is missing its runtime value"
        )
    dtype = context.numba_type(runtime_args[cursor])
    if dtype is None:
        dtype = context.dtype(runtime_args[cursor])
    if dtype is None:
        return
    try:
        _validate_runtime_integer_dtype(
            dtype,
            operation="scan",
            parameter="valid_items",
        )
    except TypeError as exc:
        raise CoopSinglePhaseRewriteError(str(exc)) from exc


__all__ = ["infer_scan_payload", "validate_scan_runtime_controls"]
