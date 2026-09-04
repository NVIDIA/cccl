# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Block and physical or logical Warp Exchange IR planning."""

from __future__ import annotations

from numba_cuda_mlir import types

from cuda.coop._core import (
    BlockExchangeMode,
    BlockExchangeValueForm,
    GroupExchangeSemantics,
    GroupLoweringPlan,
    GroupLoweringTarget,
    make_block_exchange_semantics,
    make_group_primitive_call,
    plan_group_primitive,
)

from ._group_planner_support import (
    _PAYLOAD_DTYPE_LIKE,
    Any,
    GroupRewriteError,
    ThreadGroup,
    inspect,
    ir,
)
from ._group_planning import GroupPlanningContext
from ._operations import (
    GroupResultSource,
    RewriteOperationSpec,
    register_group_primitive,
    register_rewrite_operation,
)
from ._parameters import _validate_common_numeric_dtype, normalize_dtype_param
from ._rewrite_exchange import infer_exchange_payload

_BLOCK_MODES = frozenset(mode.value for mode in BlockExchangeMode)
_WARP_MODES = frozenset(
    {
        BlockExchangeMode.STRIPED_TO_BLOCKED.value,
        BlockExchangeMode.BLOCKED_TO_STRIPED.value,
        BlockExchangeMode.SCATTER_TO_STRIPED.value,
    }
)
_PORTABLE_MODES = frozenset(
    {
        BlockExchangeMode.STRIPED_TO_BLOCKED.value,
        BlockExchangeMode.BLOCKED_TO_STRIPED.value,
    }
)


def _mode_token(value: object, *, group_kind: str) -> str:
    value = getattr(value, "value", value)
    if not isinstance(value, str):
        raise TypeError(
            "cuda.coop.numba_mlir.exchange mode must be a compile-time string"
        )
    token = value.strip().lower().replace("-", "_")
    allowed = (
        _WARP_MODES if group_kind in {"warp", "threads_within_warp"} else _BLOCK_MODES
    )
    if token not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(
            "cuda.coop.numba_mlir.exchange mode for "
            f"{group_kind} groups must be one of: {choices}"
        )
    return token


def _array_extent(
    context: GroupPlanningContext,
    value: Any,
    *,
    parameter: str,
) -> int:
    if not context.is_array("exchange", value):
        raise TypeError(
            "cuda.coop.numba_mlir.exchange requires "
            f"{parameter} to be a fixed-size ThreadData or local array"
        )
    extent = context.array_extent(value)
    if extent is None:
        raise GroupRewriteError(
            "cuda.coop.numba_mlir.exchange could not infer a static "
            f"items_per_thread extent for {parameter}"
        )
    return extent


def _array_dtype(
    context: GroupPlanningContext,
    value: Any,
    *,
    parameter: str,
) -> Any:
    dtype = context.dtype(value)
    if dtype is None:
        dtype = context.payload_write_dtype(value)
    if dtype is None:
        raise GroupRewriteError(
            f"cuda.coop.numba_mlir.exchange could not infer a dtype for {parameter}"
        )
    return dtype


def _rank_dtype(dtype: Any) -> Any:
    try:
        dtype = normalize_dtype_param(dtype)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "cuda.coop.numba_mlir.exchange ranks must have a signed integer dtype"
        ) from exc
    dtype = getattr(dtype, "literal_type", dtype)
    if (
        isinstance(dtype, types.Boolean)
        or not isinstance(dtype, types.Integer)
        or not dtype.signed
    ):
        raise TypeError(
            "cuda.coop.numba_mlir.exchange ranks must have a signed integer dtype"
        )
    return dtype


def _flag_dtype(dtype: Any) -> Any:
    try:
        dtype = normalize_dtype_param(dtype)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "cuda.coop.numba_mlir.exchange valid_flags must have an integral "
            "non-bool dtype"
        ) from exc
    dtype = getattr(dtype, "literal_type", dtype)
    if isinstance(dtype, types.Boolean) or not isinstance(dtype, types.Integer):
        raise TypeError(
            "cuda.coop.numba_mlir.exchange valid_flags must have an integral "
            "non-bool dtype"
        )
    return dtype


class _ExchangePlanning:
    """Family-local Exchange semantics over the declared planning context."""

    def __init__(self, context: GroupPlanningContext) -> None:
        self._context = context

    def _validate_common_arguments(
        self,
        operation: str,
        bound: inspect.BoundArguments,
    ) -> None:
        bound.arguments["mode"] = self._context.validate_common_selector(
            operation,
            "mode",
            bound.arguments["mode"],
            _PORTABLE_MODES,
        )

    @staticmethod
    def _provider(plan: GroupLoweringPlan, primitive: Any):
        if plan.provenance is None:
            raise GroupRewriteError(
                "cuda.coop.numba_mlir.exchange requires CUB provider provenance"
            )
        provenance = plan.provenance
        if (
            plan.target is GroupLoweringTarget.CUB_BLOCK
            and provenance.header == "cub/block/block_exchange.cuh"
            and provenance.cpp_class == "cub::BlockExchange"
        ):
            from .._lowering import _exchange

            if primitive.uses_valid_flags:
                return _exchange.exchange_flagged
            if primitive.uses_ranks:
                return _exchange.exchange_ranked
            return _exchange.exchange
        if (
            plan.target is GroupLoweringTarget.CUB_WARP
            and provenance.header == "cub/warp/warp_exchange.cuh"
            and provenance.cpp_class == "cub::WarpExchange"
        ):
            from .._lowering import _exchange

            return (
                _exchange.warp_exchange_ranked
                if primitive.uses_ranks
                else _exchange.warp_exchange
            )
        raise GroupRewriteError(
            "cuda.coop.numba_mlir.exchange received unknown CUB provider "
            f"provenance {provenance.semantic_key!r}"
        )

    def _plan(
        self,
        *,
        group: ThreadGroup,
        bound: inspect.BoundArguments,
        is_common_root: bool,
    ) -> GroupLoweringPlan:
        mode = _mode_token(
            self._context.constant(bound.arguments["mode"]),
            group_kind=group.kind,
        )
        if is_common_root and mode not in _PORTABLE_MODES:
            choices = ", ".join(sorted(_PORTABLE_MODES))
            raise ValueError(
                "cuda.coop.exchange mode must be one of: "
                f"{choices}; use cuda.coop.numba_mlir for backend-qualified "
                "scatter and warp-striped modes"
            )

        warp_time_slicing_value = bound.arguments.get("warp_time_slicing", False)
        warp_time_slicing = self._context.constant(warp_time_slicing_value)
        if not isinstance(warp_time_slicing, bool):
            raise TypeError(
                "cuda.coop.numba_mlir.exchange warp_time_slicing must be a "
                "compile-time bool"
            )
        if warp_time_slicing and group.kind != "block":
            raise ValueError(
                "cuda.coop.numba_mlir.exchange warp_time_slicing applies only "
                "to block groups"
            )
        if is_common_root and warp_time_slicing:
            raise ValueError(
                "cuda.coop.exchange does not accept warp_time_slicing; use "
                "cuda.coop.numba_mlir for this backend-qualified control"
            )

        value = bound.arguments["value"]
        items_per_thread = _array_extent(
            self._context,
            value,
            parameter="value",
        )
        dtype = _validate_common_numeric_dtype(
            _array_dtype(self._context, value, parameter="value"),
            operation="exchange",
            parameter="value",
        )
        if is_common_root and not self._context.is_thread_data(
            "exchange", "value", value
        ):
            raise TypeError(
                "cuda.coop.exchange requires value to be a fixed-size "
                "ThreadData payload; use cuda.coop.numba_mlir for "
                "backend-qualified local-array payload support"
            )

        normalized_mode = BlockExchangeMode(mode)
        ranks_value = bound.arguments.get("ranks")
        valid_flags_value = bound.arguments.get("valid_flags")
        has_ranks = not self._context.is_none(ranks_value)
        has_valid_flags = not self._context.is_none(valid_flags_value)
        if normalized_mode.uses_ranks != has_ranks:
            requirement = (
                "requires" if normalized_mode.uses_ranks else "does not accept"
            )
            raise ValueError(
                f"cuda.coop.numba_mlir.exchange {mode} {requirement} ranks"
            )
        if normalized_mode.uses_valid_flags != has_valid_flags:
            requirement = (
                "requires" if normalized_mode.uses_valid_flags else "does not accept"
            )
            raise ValueError(
                f"cuda.coop.numba_mlir.exchange {mode} {requirement} valid_flags"
            )

        rank_dtype = None
        if has_ranks:
            ranks = ranks_value
            ranks_extent = _array_extent(
                self._context,
                ranks,
                parameter="ranks",
            )
            if ranks_extent != items_per_thread:
                raise ValueError(
                    "cuda.coop.numba_mlir.exchange ranks must have the same "
                    "items_per_thread extent as value"
                )
            rank_dtype = _rank_dtype(
                _array_dtype(self._context, ranks, parameter="ranks")
            )
            if is_common_root and not self._context.is_thread_data(
                "exchange", "ranks", ranks
            ):
                raise TypeError("cuda.coop.exchange requires ranks to be ThreadData")

        valid_flag_dtype = None
        if has_valid_flags:
            valid_flags = valid_flags_value
            flags_extent = _array_extent(
                self._context,
                valid_flags,
                parameter="valid_flags",
            )
            if flags_extent != items_per_thread:
                raise ValueError(
                    "cuda.coop.numba_mlir.exchange valid_flags must have the "
                    "same items_per_thread extent as value"
                )
            valid_flag_dtype = _flag_dtype(
                _array_dtype(
                    self._context,
                    valid_flags,
                    parameter="valid_flags",
                )
            )
            if is_common_root and not self._context.is_thread_data(
                "exchange", "valid_flags", valid_flags
            ):
                raise TypeError(
                    "cuda.coop.exchange requires valid_flags to be ThreadData"
                )

        semantics = GroupExchangeSemantics(
            make_block_exchange_semantics(
                dtype=dtype,
                items_per_thread=items_per_thread,
                mode=normalized_mode,
                value_form=BlockExchangeValueForm.OUT_OF_PLACE,
                warp_time_slicing=warp_time_slicing,
                rank_dtype=rank_dtype,
                valid_flag_dtype=valid_flag_dtype,
            )
        )
        return plan_group_primitive(
            make_group_primitive_call(group, semantics),
            self._context.launch,
        ).require_supported()

    def _lower_exchange(
        self,
        inst: ir.Assign,
        *,
        operation: str,
        group: ThreadGroup,
        bound: inspect.BoundArguments,
        is_common_root: bool,
    ) -> list[Any]:
        if operation != "exchange":
            raise GroupRewriteError(
                f"Exchange planner received unexpected operation {operation!r}"
            )
        plan = self._plan(
            group=group,
            bound=bound,
            is_common_root=is_common_root,
        )
        assert plan.implementation is not None
        assert plan.topology is not None
        primitive = plan.call.operation.primitive
        factory = self._provider(plan, primitive)
        block_dim = plan.participation.exact_block_dim
        assert block_dim is not None
        factory_kwargs: dict[str, Any] = {
            "dtype": primitive.dtype,
            "threads_per_block": block_dim,
            "items_per_thread": primitive.items_per_thread,
            "mode": primitive.mode.value,
        }
        if plan.target is GroupLoweringTarget.CUB_BLOCK:
            factory_kwargs["warp_time_slicing"] = primitive.warp_time_slicing
        else:
            factory_kwargs["threads_in_warp"] = plan.topology.logical_width
        if primitive.rank_dtype is not None:
            factory_kwargs["rank_dtype"] = primitive.rank_dtype
        if primitive.valid_flag_dtype is not None:
            factory_kwargs["valid_flag_dtype"] = primitive.valid_flag_dtype

        statements: list[Any] = []
        scope = inst.target.scope
        loc = inst.loc
        value = self._context.value_var(
            statements,
            scope=scope,
            loc=loc,
            stem="exchange_value",
            value=bound.arguments["value"],
        )
        result = self._context.typed_payload_like(
            statements,
            scope=scope,
            loc=loc,
            stem="exchange_result",
            prototype=value,
            is_array=True,
            dtype_policy=_PAYLOAD_DTYPE_LIKE,
            items_per_thread=primitive.items_per_thread,
        )
        runtime_args = [value, result]
        if primitive.uses_ranks:
            ranks = self._context.value_var(
                statements,
                scope=scope,
                loc=loc,
                stem="exchange_ranks",
                value=bound.arguments.get("ranks"),
            )
            if plan.target is GroupLoweringTarget.CUB_WARP:
                preserved_ranks = self._context.typed_payload_like(
                    statements,
                    scope=scope,
                    loc=loc,
                    stem="exchange_preserved_ranks",
                    prototype=ranks,
                    is_array=True,
                    dtype_policy=_PAYLOAD_DTYPE_LIKE,
                    items_per_thread=primitive.items_per_thread,
                )
                self._context.copy_array_payload(
                    statements,
                    operation="exchange",
                    source=ranks,
                    destination=preserved_ranks,
                    scope=scope,
                    loc=loc,
                    known_items_per_thread=primitive.items_per_thread,
                )
                ranks = preserved_ranks
            runtime_args.append(ranks)
        if primitive.uses_valid_flags:
            runtime_args.append(bound.arguments.get("valid_flags"))

        statements.extend(
            self._context.rewrite_call(
                inst,
                lowering_plan=plan,
                factory=factory,
                args=runtime_args,
                kwargs=factory_kwargs,
                return_alias=result,
            )
        )
        return statements


def _lower_registered_exchange(
    context: GroupPlanningContext,
    *args: Any,
    **kwargs: Any,
) -> list[Any]:
    return _ExchangePlanning(context)._lower_exchange(*args, **kwargs)


def _validate_registered_common_arguments(
    context: GroupPlanningContext,
    operation: str,
    bound: inspect.BoundArguments,
) -> None:
    _ExchangePlanning(context)._validate_common_arguments(operation, bound)


register_group_primitive(
    "exchange",
    lower=_lower_registered_exchange,
    results=(GroupResultSource("value", "value"),),
    validate_common_arguments=_validate_registered_common_arguments,
)

_REWRITE_KWARGS = frozenset(
    {
        "dtype",
        "items_per_thread",
        "mode",
        "rank_dtype",
        "threads_in_warp",
        "threads_per_block",
        "valid_flag_dtype",
        "warp_time_slicing",
    }
)
for _operation, _namespaces, _runtime_arg_count in (
    ("exchange", frozenset({"block", "warp"}), 2),
    ("exchange_ranked", frozenset({"block", "warp"}), 3),
    ("exchange_flagged", frozenset({"block"}), 4),
):
    register_rewrite_operation(
        _operation,
        RewriteOperationSpec(
            factory_namespaces=_namespaces,
            dtype_factory_kwargs=frozenset({"dtype", "rank_dtype", "valid_flag_dtype"}),
            runtime_arg_counts=frozenset({_runtime_arg_count}),
            runtime_factory_kwargs=(),
            runtime_factory_kw_prerequisites=(),
            allowed_factory_kwargs=_REWRITE_KWARGS,
            required_factory_kwargs=frozenset(
                {"dtype", "items_per_thread", "threads_per_block"}
            ),
            accepts_temp_storage=False,
            scalar_binding_kwargs=frozenset(),
            runtime_offset_kwarg=None,
            infer_payload=infer_exchange_payload,
        ),
    )
del _namespaces, _operation, _runtime_arg_count


__all__: tuple[str, ...] = ()
