# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Block Load/Store payload inference and pre-provider validation."""

from dataclasses import dataclass
from numbers import Integral

from cuda.coop._core import ArgumentBinding, BindingKind

from ._group_rewriting import GroupRewriteContext
from ._rewrite_payload import PayloadInference
from ._rewrite_support import (
    _GLOBAL_NAME_COUNTER,
    _UNRESOLVED,
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
    def _validate_oob_default(
        context: GroupRewriteContext,
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
            from ._parameters import coerce_static_scalar

            payload_dtype = factory_kwargs.get("dtype")
            if payload_dtype is None:
                raise CoopSinglePhaseRewriteError(
                    "cuda.coop.numba_mlir.load requires an inferred dtype "
                    "before validating oob_default"
                )
            provenance = context.static_scalar_provenance(binding.value)
            source_dtype = (
                None
                if provenance is _UNRESOLVED or provenance is None
                else provenance.dtype
            )
            try:
                normalized = coerce_static_scalar(
                    binding.value,
                    payload_dtype,
                    operation="load",
                    parameter="oob_default",
                    source_dtype=source_dtype,
                )
            except (TypeError, ValueError) as exc:
                raise CoopSinglePhaseRewriteError(str(exc)) from exc
            factory_kwargs["oob_default"] = ArgumentBinding.static(normalized)
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
        value_dtype = context.numba_type(value_var)
        if value_dtype is None:
            value_dtype = context.dtype(value_var)
        if value_dtype is None:
            return

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

    @staticmethod
    def _validate_load_store_runtime_controls(
        context: GroupRewriteContext,
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

        from ._parameters import _validate_runtime_integer_dtype

        for parameter, index in checks:
            if index >= len(runtime_args) or not isinstance(
                runtime_args[index], ir.Var
            ):
                raise CoopSinglePhaseRewriteError(
                    f"coop {op_name} {parameter} must be an integer"
                )
            value_type = context.numba_type(runtime_args[index])
            if value_type is None:
                value_type = context.dtype(runtime_args[index])
            if value_type is None:
                continue
            try:
                _validate_runtime_integer_dtype(
                    value_type,
                    operation=op_name,
                    parameter=parameter,
                )
            except TypeError as exc:
                raise CoopSinglePhaseRewriteError(str(exc)) from exc

        if op_name == "load":
            _LoadStoreRewrite._validate_oob_default(
                context,
                runtime_args=runtime_args,
                factory_kwargs=factory_kwargs,
            )

    @staticmethod
    def _infer_load_store_payload(
        context: GroupRewriteContext,
        inference: PayloadInference,
    ) -> None:
        payload_var, payload_spec = inference.candidate(1)
        memory_var = inference.runtime_args[0] if inference.runtime_args else None
        memory_dtype = (
            context.dtype(memory_var) if isinstance(memory_var, ir.Var) else None
        )

        payload_is_array = payload_spec is not None
        if payload_spec is None:
            payload_dtype = (
                context.dtype(payload_var) if isinstance(payload_var, ir.Var) else None
            )
            inference.infer_kwarg("items_per_thread", 1)
            inference.infer_kwarg(
                "dtype",
                memory_dtype if memory_dtype is not None else payload_dtype,
            )
        else:
            inference.infer_kwarg("items_per_thread", payload_spec.items_per_thread)
            payload_dtype = payload_spec.dtype
            if payload_dtype is None and payload_var is not None:
                payload_dtype = context.dtype(payload_var)
            if (
                inference.op_name == "store"
                and payload_dtype is None
                and payload_var is not None
            ):
                payload_dtype = context.infer_thread_data_write_dtype(payload_var)
            inferred_dtype = memory_dtype if memory_dtype is not None else payload_dtype
            if inferred_dtype is None:
                inferred_dtype = inference.factory_value("dtype")
            inference.infer_kwarg("dtype", inferred_dtype)
            if inferred_dtype is not None and payload_var is not None:
                context.record_thread_data_dtype(payload_var, inferred_dtype)

        from ._parameters import _validate_common_numeric_dtype

        if inference.op_name == "store" and not payload_is_array:
            provenance = context.static_scalar_provenance(payload_var)
            if provenance is not _UNRESOLVED and memory_dtype is not None:
                from ._parameters import coerce_static_scalar

                try:
                    coerce_static_scalar(
                        provenance.value,
                        memory_dtype,
                        operation="store",
                        parameter="value",
                        source_dtype=provenance.dtype,
                    )
                except (TypeError, ValueError) as exc:
                    raise CoopSinglePhaseRewriteError(str(exc)) from exc
                payload_dtype = memory_dtype

        provider_dtype = inference.factory_value("dtype")

        if provider_dtype is not None:
            try:
                _validate_common_numeric_dtype(
                    provider_dtype, operation=inference.op_name
                )
            except (TypeError, ValueError) as exc:
                raise CoopSinglePhaseRewriteError(str(exc)) from exc
        if payload_dtype is not None:
            try:
                _validate_common_numeric_dtype(
                    payload_dtype,
                    operation=inference.op_name,
                )
            except (TypeError, ValueError) as exc:
                raise CoopSinglePhaseRewriteError(str(exc)) from exc
        if (
            memory_dtype is not None
            and payload_dtype is not None
            and not _dtype_values_match(memory_dtype, payload_dtype)
        ):
            if inference.op_name != "store" or payload_is_array:
                raise CoopSinglePhaseRewriteError(
                    f"cuda.coop.numba_mlir.{inference.op_name} memory dtype "
                    f"{memory_dtype} does not match payload dtype {payload_dtype}"
                )
            raise CoopSinglePhaseRewriteError(
                "cuda.coop.numba_mlir.store value dtype "
                f"{payload_dtype} does not match destination dtype {memory_dtype}"
            )


def infer_load_store_payload(
    context: GroupRewriteContext,
    inference: PayloadInference,
) -> None:
    _LoadStoreRewrite._infer_load_store_payload(context, inference)


def validate_load_store_runtime_controls(
    context: GroupRewriteContext,
    *,
    op_name: str,
    runtime_args: list[ir.Var],
    factory_kwargs: dict[str, object],
) -> None:
    _LoadStoreRewrite._validate_load_store_runtime_controls(
        context,
        op_name=op_name,
        runtime_args=runtime_args,
        factory_kwargs=factory_kwargs,
    )


def analyze_load_store_match(
    context: GroupRewriteContext,
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
        operands = list(zip(operand_names, runtime_args))
        if op_name == "store" and len(runtime_args) >= 2:
            value_is_array = context.thread_data(runtime_args[1])
            if value_is_array is None:
                operands = operands[:1]
        for operand_name, operand in operands:
            operand_dtype = context.dtype(operand)
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
            group_root_store and context.thread_data(runtime_args[1]) is None
        )
    )


def prepare_load_store_runtime_args(
    context: GroupRewriteContext,
    block: ir.Block,
    *,
    match: _RewriteMatch,
    runtime_args: list[ir.Var],
    scope: ir.Scope,
    loc: ir.Loc,
) -> list[ir.Var]:
    del context
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
