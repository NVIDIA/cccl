# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Lower public Exchange calls to private Numba-CUDA-MLIR providers.

Recover static mode and per-thread payload facts before ordinary typing. Ask
the shared group planner for a supported CUB implementation, select its
registered provider, and build replacement IR with a fresh result payload.
Provider rewriting later allocates that payload and supplies shared scratch.
Common-API validation retains the common layout subset. Qualified block
calls also accept scatter and warp-striped modes. Their contracts are
documented by ``cuda.coop.numba_mlir.exchange``.
"""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any

import numba_cuda_mlir.numba_cuda.types as numba_types

from cuda.coop._core import (
    BlockExchangeMode,
    BlockExchangeValueForm,
    GroupExchangeSemantics,
    GroupLoweringPlan,
    GroupLoweringTarget,
    ThreadGroup,
    make_block_exchange_semantics,
    make_group_primitive_call,
    plan_group_primitive,
)

from ._group_planner_support import _PAYLOAD_DTYPE_LIKE, GroupRewriteError, ir
from ._group_planning import GroupPlanningContext
from ._operations import (
    GroupResultSource,
    RewriteOperationSpecification,
    register_group_primitive,
    register_rewrite_operation,
)
from ._parameters import _validate_common_numeric_dtype, normalize_dtype_param
from ._rewrite_exchange import infer_exchange_payload

_BLOCK_MODES = frozenset(mode.value for mode in BlockExchangeMode)
_COMMON_MODES = frozenset(
    {
        BlockExchangeMode.STRIPED_TO_BLOCKED.value,
        BlockExchangeMode.BLOCKED_TO_STRIPED.value,
    }
)
_WARP_MODES = _COMMON_MODES


def _mode_token(value: object, *, group_kind: str) -> str:
    if not isinstance(value, str) or isinstance(value, Enum):
        raise TypeError(
            "cuda.coop.numba_mlir.exchange mode must be a compile-time string"
        )
    token = value.strip().lower().replace("-", "_")
    allowed = (
        _WARP_MODES
        if group_kind in {"warp", "threads_within_warp"}
        else _BLOCK_MODES
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
    """Require a recognized array with a known per-thread item count.

    Exchange must select a fixed-size CUB overload before normal typing.
    Report the operand name when its origin or extent cannot establish that
    shape; scalar operands are not accepted as one-item arrays.

    Parameters
    ----------
    context : GroupPlanningContext
        Access to launch dimensions, constant controls, payload
        facts, and IR builders for this group-planning attempt.
    value : ir.Var
        Input, rank, or validity payload to inspect.
    parameter : str
        Public argument name included in diagnostics.
    """

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
    """Infer an array element type from provenance or known payload writes.

    The write fallback supports a payload whose constructor omitted dtype.
    Raise a planning error if neither source supplies the element type.
    """

    dtype = context.dtype(value)
    if dtype is None:
        dtype = context.payload_write_dtype(value)
    if dtype is None:
        raise GroupRewriteError(
            f"cuda.coop.numba_mlir.exchange could "
            f"not infer a dtype for {parameter}"
        )
    return dtype


def _rank_dtype(dtype: Any) -> Any:
    try:
        dtype = normalize_dtype_param(dtype)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "cuda.coop.numba_mlir.exchange ranks "
            "must have a signed integer dtype"
        ) from exc
    dtype = getattr(dtype, "literal_type", dtype)
    if (
        isinstance(dtype, numba_types.Boolean)
        or not isinstance(dtype, numba_types.Integer)
        or not dtype.signed
    ):
        raise TypeError(
            "cuda.coop.numba_mlir.exchange ranks "
            "must have a signed integer dtype"
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
    if isinstance(dtype, numba_types.Boolean) or not isinstance(
        dtype, numba_types.Integer
    ):
        raise TypeError(
            "cuda.coop.numba_mlir.exchange valid_flags must have an integral "
            "non-bool dtype"
        )
    return dtype


class _ExchangePlanning:
    """Validate one Exchange call and construct its provider replacement.

    The shared context supplies launch facts, argument provenance, and IR
    builders. Planning selects a core contract first; lowering then chooses a
    matching registered factory and creates the public result payload. The
    class returns statements for the owning planner to install.
    """

    def __init__(self, context: GroupPlanningContext) -> None:
        self._context = context

    def _validate_common_arguments(
        self,
        operation: str,
        bound: inspect.BoundArguments,
    ) -> None:
        """Normalize the mode and enforce the common API's two layout choices.

        Update the bound mode in place. Backend-only scatter and warp-striped
        modes must be requested through the qualified Numba-CUDA-MLIR API.
        """

        bound.arguments["mode"] = self._context.validate_common_selector(
            operation,
            "mode",
            bound.arguments["mode"],
            _COMMON_MODES,
        )

    @staticmethod
    def _provider(plan: GroupLoweringPlan, primitive: Any):
        """Match the core plan to a registered Exchange provider factory.

        Check the target, native header, and C++ class together. Then select
        the plain, ranked, or flagged ABI from the semantic operands. Raise a
        planning error if the provenance has no supported provider; a matching
        operation name alone is not enough to choose an implementation.
        """

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
        """Validate Exchange arguments and request a supported core plan.

        Resolve mode and time slicing as constants. Require a fixed numeric
        array value, plus same-extent rank or validity arrays when the mode
        requires them. Ranks use signed integers; flags use non-boolean
        integer types. Common API operands must originate from ThreadData.

        Build separate-input/output semantics and resolve them against the
        group and launch. Return the supported plan. The caller must supply
        valid ranks and unique destinations where the mode requires them,
        and must initialize unwritten output positions before reading them.
        """

        mode = _mode_token(
            self._context.constant(bound.arguments["mode"]),
            group_kind=group.kind,
        )
        if is_common_root and mode not in _COMMON_MODES:
            choices = ", ".join(sorted(_COMMON_MODES))
            raise ValueError(
                "cuda.coop.exchange mode must be one of: "
                f"{choices}; use cuda.coop.numba_mlir for backend-qualified "
                "scatter and warp-striped modes"
            )

        warp_time_slicing_value = bound.arguments.get(
            "warp_time_slicing", False
        )
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
                "requires"
                if normalized_mode.uses_valid_flags
                else "does not accept"
            )
            raise ValueError(
                f"cuda.coop.numba_mlir.exchange "
                f"{mode} {requirement} valid_flags"
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
                raise TypeError(
                    "cuda.coop.exchange requires ranks to be ThreadData"
                )

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

        from .._lowering._core import NumbaMlirCoreAdapter

        adapter = NumbaMlirCoreAdapter()
        semantics = GroupExchangeSemantics(
            make_block_exchange_semantics(
                dtype=adapter.core_dtype(dtype),
                items_per_thread=items_per_thread,
                mode=normalized_mode,
                value_form=BlockExchangeValueForm.OUT_OF_PLACE,
                warp_time_slicing=warp_time_slicing,
                rank_dtype=adapter.core_dtype(rank_dtype),
                valid_flag_dtype=adapter.core_dtype(valid_flag_dtype),
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
        """Build an Exchange provider call that returns a fresh payload.

        Validate the call, choose the provider, and bind its specialization
        keywords from the plan. Emit a result marker with the input dtype and
        extent; pass it as the provider's output and alias it to the original
        public result after the provider runs.

        CUB warp scatter can modify its rank array, so a warp scatter
        plan copies ranks first. Current mode validation admits only layout
        conversions for warp groups; this rank copy is defensive.

        Return ordered replacement statements. The owning planner installs
        them after whole-function validation.
        """

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
        semantics = plan.call.operation
        assert isinstance(semantics, GroupExchangeSemantics)
        assert plan.participation is not None
        primitive = semantics.primitive
        factory = self._provider(plan, primitive)
        block_dim = plan.participation.exact_block_dim
        assert block_dim is not None
        from .._lowering._core import NumbaMlirCoreAdapter

        adapter = NumbaMlirCoreAdapter()
        factory_kwargs: dict[str, Any] = {
            "dtype": adapter.normalize_dtype(primitive.dtype),
            "threads_per_block": block_dim,
            "items_per_thread": primitive.items_per_thread,
            "mode": primitive.mode.value,
        }
        if plan.target is GroupLoweringTarget.CUB_BLOCK:
            factory_kwargs["warp_time_slicing"] = primitive.warp_time_slicing
        else:
            factory_kwargs["threads_in_warp"] = plan.topology.logical_width
        if primitive.rank_dtype is not None:
            factory_kwargs["rank_dtype"] = adapter.normalize_dtype(
                primitive.rank_dtype
            )
        if primitive.valid_flag_dtype is not None:
            factory_kwargs["valid_flag_dtype"] = adapter.normalize_dtype(
                primitive.valid_flag_dtype
            )

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
            runtime_args.append(bound.arguments["valid_flags"])

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
        RewriteOperationSpecification(
            factory_namespaces=_namespaces,
            dtype_factory_kwargs=frozenset(
                {"dtype", "rank_dtype", "valid_flag_dtype"}
            ),
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
