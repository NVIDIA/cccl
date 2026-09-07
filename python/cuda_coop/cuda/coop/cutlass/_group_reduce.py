# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CUTLASS group-first reduction entrypoint."""

from __future__ import annotations

from numbers import Integral
from typing import Any

from cuda.coop._core import (
    ArgumentBinding,
    CxxOperator,
    Dependency,
    GroupLoweringPlan,
    GroupReduceSemantics,
    LaunchFacts,
    ReduceOperation,
    ReduceValueKind,
    make_group_primitive_call,
    make_reduce_semantics,
    plan_group_primitive,
)
from cuda.coop._core.block import BlockReduceAlgorithm

from ._internal._thread_data import _coerce_thread_payload
from ._thread_group import ThreadGroup

_SCOPE = __name__.rsplit(".", 1)[0]

_REDUCE_OPERATOR_CPP = {
    "multiplies": "::cuda::std::multiplies<T>",
    "min": "::cuda::minimum<T>",
    "max": "::cuda::maximum<T>",
    "bit_and": "::cuda::std::bit_and<T>",
    "bit_or": "::cuda::std::bit_or<T>",
    "bit_xor": "::cuda::std::bit_xor<T>",
}

_BLOCK_REDUCE_ALGORITHM_ALIASES = {
    "raking_commutative_only": BlockReduceAlgorithm.RAKING_COMMUTATIVE_ONLY,
    "raking": BlockReduceAlgorithm.RAKING,
    "warp_reductions": BlockReduceAlgorithm.WARP_REDUCTIONS,
    "warp_reductions_nondeterministic": (
        BlockReduceAlgorithm.WARP_REDUCTIONS_NONDETERMINISTIC
    ),
}


def _normalize_reduce_op(binary_op: Any) -> str:
    from ._dsl._scope import normalize_reduce_op

    return normalize_reduce_op(binary_op, scope=_SCOPE)


def _classify_valid_items(valid_items: Any) -> ArgumentBinding:
    if valid_items is None:
        return ArgumentBinding.omitted()
    if isinstance(valid_items, bool):
        raise TypeError("valid_items must be an integer, not bool")
    if isinstance(valid_items, Integral):
        return ArgumentBinding.static(int(valid_items))
    return ArgumentBinding.runtime()


def _normalize_block_reduce_algorithm(algorithm: Any) -> Any:
    if isinstance(algorithm, str):
        return _BLOCK_REDUCE_ALGORITHM_ALIASES.get(algorithm, algorithm)
    return algorithm


def _validate_group_for_reduce(group: ThreadGroup) -> None:
    """Require the explicit execution-group descriptor."""

    if not isinstance(group, ThreadGroup):
        raise TypeError(f"{_SCOPE}.reduce group must be a ThreadGroup")


def _resolve_group_for_reduce(
    group: ThreadGroup,
    launch_kwargs: dict[str, Any],
) -> ThreadGroup:
    """Compatibility helper that resolves the active block shape.

    New calls go straight through ``plan_group_primitive``; this remains only
    for existing internal API users during the namespace migration.
    """

    from ._thread_group import _resolve_collective_group

    return _resolve_collective_group(
        group,
        launch_kwargs,
        feature="reduce",
    )


def _make_group_reduce_plan(
    *,
    group: ThreadGroup,
    launch: LaunchFacts,
    dtype: Any,
    value_kind: ReduceValueKind,
    items_per_thread: int,
    op: str,
    broadcast: bool,
    valid_items: ArgumentBinding | None = None,
    algorithm: Any = None,
    source: str = "cutlass_root",
) -> GroupLoweringPlan:
    """Build the canonical shared-core plan for one CUTLASS group reduction."""

    reduce_operator = None
    operation = ReduceOperation.SUM
    if op != "sum":
        try:
            cpp = _REDUCE_OPERATOR_CPP[op]
        except KeyError as exc:
            raise NotImplementedError(
                f"unsupported group reduce operator {op!r}"
            ) from exc
        operation = ReduceOperation.REDUCE
        reduce_operator = CxxOperator(
            cpp=cpp,
            dtype=Dependency("T"),
            name="binary_op",
        )
    if valid_items is None:
        valid_items = ArgumentBinding.omitted()
    primitive = make_reduce_semantics(
        dtype=dtype,
        operation=operation,
        value_kind=value_kind,
        items_per_thread=items_per_thread,
        reduce_operator=reduce_operator,
        valid_items=valid_items,
    )
    call = make_group_primitive_call(
        group,
        GroupReduceSemantics(
            primitive=primitive,
            broadcast=broadcast,
            cub_algorithm=algorithm,
        ),
        source=source,
    )
    return plan_group_primitive(call, launch)


def _reduce(
    group: ThreadGroup,
    value: Any,
    /,
    *args: Any,
    binary_op: Any = None,
    broadcast: bool = True,
    valid_items: Any = None,
    algorithm: Any = None,
    **kwargs: Any,
) -> Any:
    """Reduce values across a CUDA thread group with CUDAX or direct CUB.

    ``group`` always names the participating threads explicitly.
    Full-group reductions use CUDAX and may be broadcast to every member or
    returned only at rank zero. ``valid_items`` and explicit block ``algorithm``
    selectors use direct CUB and therefore require ``broadcast=False``.

    With ``broadcast=False``, only group rank zero has a defined result; other
    members receive an implementation placeholder and must not consume it.
    Every group member must still invoke the collective in converged control
    flow; root-only result ownership never means root-only participation.
    ``valid_items`` counts the first contributing group members, not the total
    items inside ``ThreadData``, and is therefore supported only for scalar
    operands. It changes which members contribute, not which members call the
    collective. Its valid range is ``1 <= valid_items <= group.static_size``.
    Static counts are checked while tracing. A runtime count is a uniform caller
    precondition and is passed directly to CUB without a device-side range
    check.

    Register-resident CuTe tensor fragments and ``TensorSSA`` values are
    adapted to ``ThreadData`` directly without another global/shared-memory
    load or bulk copy. Values that are already thread-local do not require an
    intervening ``coop.load`` call.

    Supported block algorithm names are ``raking_commutative_only``, ``raking``,
    and ``warp_reductions``. The nondeterministic CUB specialization is excluded
    because its current implementation is addition-specific. All direct CUB
    routes return a value defined only at the group root.
    """

    from ._dsl._launch import infer_launch_facts, pop_launch_metadata
    from ._dsl._scope import merge_payload, validate_no_extra_args

    payload = merge_payload(
        _SCOPE,
        "reduce",
        {
            "group": group,
            "value": value,
            "args": args,
            "binary_op": binary_op,
            "broadcast": broadcast,
            "valid_items": valid_items,
            "algorithm": algorithm,
        },
        kwargs,
    )
    launch_kwargs = pop_launch_metadata(kwargs)
    validate_no_extra_args(
        _SCOPE,
        "reduce",
        args=payload.pop("args"),
        kwargs=kwargs,
        expected=(
            "expects a ThreadGroup and one positional value, with optional "
            "valid_items and block algorithm selectors"
        ),
    )
    _validate_group_for_reduce(group)
    if not isinstance(broadcast, bool):
        raise TypeError(f"{_SCOPE}.reduce broadcast must be a bool")
    op = _normalize_reduce_op(binary_op)
    valid_items_binding = _classify_valid_items(valid_items)
    algorithm = _normalize_block_reduce_algorithm(algorithm)
    value = _coerce_thread_payload(
        value,
        scope=_SCOPE,
        primitive_name="reduce",
        arg_name="value",
        common_root_payload_kind="scalar_or_thread_data",
    )
    launch = infer_launch_facts(
        launch_kwargs,
        scope=_SCOPE,
        primitive_name="reduce",
    )
    from ._dsl import _cudax_reduce_provider as _provider

    return _provider.provider_reduce(
        group=group,
        launch=launch,
        value=value,
        op=op,
        broadcast=broadcast,
        valid_items=valid_items,
        valid_items_binding=valid_items_binding,
        algorithm=algorithm,
    )


def reduce(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    binary_op: Any = None,
    broadcast: bool = True,
    valid_items: Any = None,
    algorithm: Any = None,
) -> Any:
    """Reduce values across a CUDA thread group with CUDAX or direct CUB."""

    return _reduce(
        group,
        value,
        binary_op=binary_op,
        broadcast=broadcast,
        valid_items=valid_items,
        algorithm=algorithm,
    )


def sum(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    broadcast: bool = True,
    valid_items: Any = None,
    algorithm: Any = None,
) -> Any:
    """Sum values across a CUDA thread group."""

    return _reduce(
        group,
        value,
        binary_op=None,
        broadcast=broadcast,
        valid_items=valid_items,
        algorithm=algorithm,
    )


__all__ = ["reduce", "sum"]
