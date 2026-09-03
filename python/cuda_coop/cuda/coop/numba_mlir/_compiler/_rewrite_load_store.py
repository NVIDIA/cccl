# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Block Load/Store payload inference and pre-provider validation."""

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np

from cuda.coop._core import ArgumentBinding, BindingKind

from ._rewrite_payload import PayloadInference
from ._rewrite_support import (
    _GLOBAL_NAME_COUNTER,
    CoopSinglePhaseRewriteError,
    _cuda_module,
    _dtype_values_match,
    _next_global_name,
    _RewriteMatch,
    ir,
)


@dataclass(frozen=True)
class _LoadStoreMatchMetadata:
    box_root_store_scalar: bool = False


class _LoadStoreRewrite:
    @staticmethod
    def _validate_static_oob_default(value: object) -> None:
        if isinstance(value, np.generic):
            value_dtype = value.dtype
        elif type(value) in {bool, int, float, complex}:
            value_dtype = type(value)
        else:
            raise CoopSinglePhaseRewriteError(
                "cuda.coop.numba_mlir.load static oob_default must be a "
                "portable numeric scalar"
            )

        from ._parameters import _validate_common_numeric_dtype

        try:
            _validate_common_numeric_dtype(
                value_dtype,
                operation="load",
                parameter="oob_default",
            )
        except (TypeError, ValueError) as exc:
            raise CoopSinglePhaseRewriteError(str(exc)) from exc

        if isinstance(value, Integral):
            normalized = int(value)
            if not -(1 << 63) <= normalized <= (1 << 64) - 1:
                raise CoopSinglePhaseRewriteError(
                    "cuda.coop.numba_mlir.load static oob_default must fit a "
                    "64-bit integer"
                )
        elif isinstance(value, Real) and not math.isfinite(float(value)):
            raise CoopSinglePhaseRewriteError(
                "cuda.coop.numba_mlir.load static oob_default must be finite"
            )

    def _validate_oob_default(
        self,
        *,
        runtime_args: list[ir.Var],
        factory_kwargs: dict[str, object],
    ) -> None:
        binding = factory_kwargs.get("oob_default")
        if not isinstance(binding, ArgumentBinding) or (
            binding.kind is BindingKind.OMITTED
        ):
            return
        if binding.kind is BindingKind.STATIC:
            self._validate_static_oob_default(binding.value)
            return

        runtime_index = 2
        valid_items = factory_kwargs.get("num_valid_items")
        if (
            isinstance(valid_items, ArgumentBinding)
            and valid_items.kind is BindingKind.RUNTIME
        ):
            runtime_index += 1
        if runtime_index >= len(runtime_args) or not isinstance(
            runtime_args[runtime_index], ir.Var
        ):
            raise CoopSinglePhaseRewriteError(
                "cuda.coop.numba_mlir.load runtime oob_default is missing its "
                "runtime value"
            )

        value_var = runtime_args[runtime_index]
        value_dtype = self._resolve_var_numba_type(value_var)
        if value_dtype is None:
            value_dtype = self._resolve_var_dtype(value_var)
        if value_dtype is None:
            raise CoopSinglePhaseRewriteError(
                "cuda.coop.numba_mlir.load could not infer the runtime "
                "oob_default dtype before provider materialization"
            )

        from ._parameters import _validate_common_numeric_dtype

        try:
            value_dtype = _validate_common_numeric_dtype(
                value_dtype,
                operation="load",
                parameter="oob_default",
            )
        except (TypeError, ValueError) as exc:
            raise CoopSinglePhaseRewriteError(str(exc)) from exc
        payload_dtype = factory_kwargs.get("dtype")
        if payload_dtype is not None and not _dtype_values_match(
            value_dtype,
            payload_dtype,
        ):
            raise CoopSinglePhaseRewriteError(
                "cuda.coop.numba_mlir.load runtime oob_default dtype "
                f"{value_dtype} does not match payload dtype {payload_dtype}"
            )

    def _validate_load_store_runtime_controls(
        self,
        *,
        op_name: str,
        runtime_args: list[ir.Var],
        factory_kwargs: dict[str, object],
    ) -> None:
        if op_name not in {"load", "store"}:
            raise CoopSinglePhaseRewriteError(
                f"unsupported Numba-CUDA-MLIR operation {op_name!r}"
            )

        valid_items = factory_kwargs.get("num_valid_items")
        if (
            isinstance(valid_items, ArgumentBinding)
            and valid_items.kind is BindingKind.STATIC
            and (
                isinstance(valid_items.value, bool)
                or not isinstance(valid_items.value, Integral)
            )
        ):
            raise CoopSinglePhaseRewriteError(
                f"coop {op_name} valid_items must be an integer"
            )

        offset = factory_kwargs.get("offset")
        if isinstance(offset, ArgumentBinding) and offset.kind is BindingKind.STATIC:
            if isinstance(offset.value, bool) or not isinstance(offset.value, Integral):
                raise CoopSinglePhaseRewriteError(
                    f"coop {op_name} offset must be an integer"
                )
            normalized_offset = int(offset.value)
            if normalized_offset < 0:
                raise CoopSinglePhaseRewriteError(
                    "coop load/store static offset must be nonnegative"
                )
            if normalized_offset > (1 << 63) - 1:
                raise CoopSinglePhaseRewriteError(
                    "coop load/store static offset must fit a signed 64-bit integer"
                )

        checks: list[tuple[str, int]] = []
        cursor = 2
        if (
            isinstance(valid_items, ArgumentBinding)
            and valid_items.kind is BindingKind.RUNTIME
        ):
            checks.append(("valid_items", cursor))
            cursor += 1
        oob_default = factory_kwargs.get("oob_default")
        if (
            isinstance(oob_default, ArgumentBinding)
            and oob_default.kind is BindingKind.RUNTIME
        ):
            cursor += 1
        if len(runtime_args) > cursor:
            checks.append(("offset", cursor))

        from numba_cuda_mlir import types as numba_mlir_types

        for parameter, index in checks:
            if index >= len(runtime_args) or not isinstance(
                runtime_args[index], ir.Var
            ):
                raise CoopSinglePhaseRewriteError(
                    f"coop {op_name} {parameter} must be an integer"
                )
            value_type = self._resolve_var_numba_type(runtime_args[index])
            if value_type is None:
                value_type = self._resolve_var_dtype(runtime_args[index])
            if isinstance(value_type, numba_mlir_types.Boolean) or not isinstance(
                value_type, numba_mlir_types.Integer
            ):
                raise CoopSinglePhaseRewriteError(
                    f"coop {op_name} {parameter} must be an integer, not bool "
                    "or a noninteger scalar"
                )

        if op_name == "load":
            self._validate_oob_default(
                runtime_args=runtime_args,
                factory_kwargs=factory_kwargs,
            )

    def _infer_load_store_payload(self, inference: PayloadInference) -> None:
        payload_var, payload_spec = inference.candidate(1)
        memory_var = inference.runtime_args[0] if inference.runtime_args else None
        memory_dtype = (
            self._resolve_var_dtype(memory_var)
            if isinstance(memory_var, ir.Var)
            else None
        )

        if payload_spec is None:
            payload_dtype = (
                self._resolve_var_dtype(payload_var)
                if isinstance(payload_var, ir.Var)
                else None
            )
            if inference.op_name == "store":
                if payload_dtype is None and payload_var is not None:
                    payload_dtype = self._infer_thread_data_dtype_from_writes(
                        payload_var
                    )
                inference.infer_kwarg("items_per_thread", 1)
                inference.infer_kwarg(
                    "dtype",
                    payload_dtype if payload_dtype is not None else memory_dtype,
                )
        else:
            inference.infer_kwarg("items_per_thread", payload_spec.items_per_thread)
            payload_dtype = payload_spec.dtype
            if payload_dtype is None and payload_var is not None:
                payload_dtype = self._resolve_var_dtype(payload_var)
            if (
                inference.op_name == "store"
                and payload_dtype is None
                and payload_var is not None
            ):
                payload_dtype = self._infer_thread_data_dtype_from_writes(payload_var)
            inferred_dtype = (
                payload_dtype if payload_dtype is not None else memory_dtype
            )
            if inferred_dtype is None:
                inferred_dtype = inference.factory_value("dtype")
            inference.infer_kwarg("dtype", inferred_dtype)
            if inferred_dtype is not None and payload_var is not None:
                self._record_inferred_thread_data_dtype(payload_var, inferred_dtype)

        provider_dtype = inference.factory_value("dtype")
        from ._parameters import _validate_common_numeric_dtype

        if provider_dtype is not None:
            try:
                _validate_common_numeric_dtype(
                    provider_dtype, operation=inference.op_name
                )
            except (TypeError, ValueError) as exc:
                raise CoopSinglePhaseRewriteError(str(exc)) from exc
        if (
            memory_dtype is not None
            and payload_dtype is not None
            and not _dtype_values_match(memory_dtype, payload_dtype)
        ):
            raise CoopSinglePhaseRewriteError(
                f"cuda.coop.numba_mlir.{inference.op_name} memory dtype "
                f"{memory_dtype} does not match payload dtype {payload_dtype}"
            )


class _LoadStoreRewriteHandler(_LoadStoreRewrite):
    """Family-local rewrite facade over the shared IR analysis context."""

    def __init__(self, rewrite: Any) -> None:
        self._rewrite = rewrite

    def __getattr__(self, name: str) -> Any:
        return getattr(self._rewrite, name)


def infer_load_store_payload(rewrite: Any, inference: PayloadInference) -> None:
    _LoadStoreRewriteHandler(rewrite)._infer_load_store_payload(inference)


def validate_load_store_runtime_controls(
    rewrite: Any,
    *,
    op_name: str,
    runtime_args: list[ir.Var],
    factory_kwargs: dict[str, object],
) -> None:
    _LoadStoreRewriteHandler(rewrite)._validate_load_store_runtime_controls(
        op_name=op_name,
        runtime_args=runtime_args,
        factory_kwargs=factory_kwargs,
    )


def analyze_load_store_match(
    rewrite: Any,
    *,
    op_name: str,
    runtime_args: tuple[ir.Var, ...],
    factory_kwargs: dict[str, object],
) -> _LoadStoreMatchMetadata:
    group_root_store = factory_kwargs.pop("_group_root_store", False)
    common_root_operation = factory_kwargs.pop("_common_root_operation", None)
    if not isinstance(group_root_store, bool):
        raise CoopSinglePhaseRewriteError(
            "_group_root_store must be a compile-time bool"
        )
    if common_root_operation is not None:
        if common_root_operation != op_name:
            raise CoopSinglePhaseRewriteError(
                "_common_root_operation does not match the rewritten group operation"
            )
        from ._parameters import _validate_common_numeric_dtype

        operand_names = (
            ("source", "output") if op_name == "load" else ("destination", "value")
        )
        for operand_name, operand in zip(operand_names, runtime_args):
            operand_dtype = rewrite._resolve_var_dtype(operand)
            if operand_dtype is None:
                raise CoopSinglePhaseRewriteError(
                    f"Failed to infer cuda.coop.{common_root_operation} "
                    f"{operand_name} dtype for portable API validation."
                )
            try:
                _validate_common_numeric_dtype(
                    operand_dtype,
                    operation=common_root_operation,
                )
            except (TypeError, ValueError) as exc:
                raise CoopSinglePhaseRewriteError(str(exc)) from exc
    if group_root_store and (op_name != "store" or len(runtime_args) < 2):
        raise CoopSinglePhaseRewriteError(
            "_group_root_store is valid only for root store calls"
        )
    return _LoadStoreMatchMetadata(
        box_root_store_scalar=(
            group_root_store
            and rewrite._resolve_thread_data_spec(runtime_args[1]) is None
        )
    )


def prepare_load_store_runtime_args(
    rewrite: Any,
    block: ir.Block,
    *,
    match: _RewriteMatch,
    runtime_args: list[ir.Var],
    scope: ir.Scope,
    loc: ir.Loc,
) -> list[ir.Var]:
    del rewrite
    metadata = match.family_metadata
    if not isinstance(metadata, _LoadStoreMatchMetadata):
        raise CoopSinglePhaseRewriteError("missing Load/Store family metadata")
    if not metadata.box_root_store_scalar:
        return runtime_args
    if len(runtime_args) < 2:
        raise CoopSinglePhaseRewriteError("root store is missing its value argument")
    dtype = match.factory_kwargs.get("dtype")
    if dtype is None:
        raise CoopSinglePhaseRewriteError("root store requires an inferred dtype")
    items_per_thread = match.factory_kwargs.get("items_per_thread", 1)
    if (
        isinstance(items_per_thread, bool)
        or not isinstance(items_per_thread, int)
        or items_per_thread < 1
    ):
        raise CoopSinglePhaseRewriteError(
            "root store requires an inferred positive items_per_thread"
        )

    def new_var(stem: str) -> ir.Var:
        return ir.Var(
            scope,
            f"__coop_root_store_{stem}_{next(_GLOBAL_NAME_COUNTER)}__",
            loc,
        )

    module_var = new_var("cuda")
    local_var = new_var("local")
    array_fn = new_var("array")
    shape_var = new_var("shape")
    dtype_var = new_var("dtype")
    payload = new_var("payload")
    block.append(
        ir.Assign(
            ir.Global(_next_global_name("root_store_cuda"), _cuda_module, loc),
            module_var,
            loc,
        )
    )
    block.append(ir.Assign(ir.Expr.getattr(module_var, "local", loc), local_var, loc))
    block.append(ir.Assign(ir.Expr.getattr(local_var, "array", loc), array_fn, loc))
    block.append(ir.Assign(ir.Const(items_per_thread, loc), shape_var, loc))
    block.append(
        ir.Assign(
            ir.Global(_next_global_name("root_store_dtype"), dtype, loc),
            dtype_var,
            loc,
        )
    )
    block.append(
        ir.Assign(
            ir.Expr.call(array_fn, [shape_var, dtype_var], (), loc),
            payload,
            loc,
        )
    )
    value = runtime_args[1]
    for item_index in range(items_per_thread):
        index_var = new_var(f"index_{item_index}")
        block.append(ir.Assign(ir.Const(item_index, loc), index_var, loc))
        block.append(ir.SetItem(payload, index_var, value, loc))
    runtime_args[1] = payload
    return runtime_args


__all__ = [
    "analyze_load_store_match",
    "infer_load_store_payload",
    "prepare_load_store_runtime_args",
    "validate_load_store_runtime_controls",
]
