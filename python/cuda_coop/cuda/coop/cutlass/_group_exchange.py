# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CUTLASS group-first exchange entrypoint."""

from __future__ import annotations

from typing import Any

from cuda.coop._core import (
    GroupExchangeMode,
    GroupExchangeSemantics,
    GroupLoweringPlan,
    LaunchFacts,
    make_group_primitive_call,
    plan_group_primitive,
)
from cuda.coop._core.block import make_block_exchange_semantics

from ._internal import ThreadData
from ._internal._thread_data import _coerce_thread_payload
from ._limits import MAX_GROUP_EXCHANGE_ITEMS_PER_THREAD
from ._thread_group import (
    ThreadGroup,
    _require_complete_warp_partition,
    _resolve_collective_group,
    _resolve_collective_group_from_launch,
)

_SCOPE = __name__.rsplit(".", 1)[0]


def _normalize_exchange_mode(mode: Any) -> str:
    try:
        normalized = GroupExchangeMode(mode).value
    except (TypeError, ValueError) as exc:
        choices = ", ".join(item.value for item in GroupExchangeMode)
        raise ValueError(f"{_SCOPE}.exchange mode must be one of: {choices}") from exc
    return normalized


def _validate_group_for_exchange(group: ThreadGroup) -> None:
    if group.kind not in {"block", "warp", "threads_within_warp"}:
        raise NotImplementedError(
            f"{_SCOPE}.exchange currently lowers CUB exchanges only for "
            "block, physical-warp, and logical-warp groups"
        )


def _make_group_exchange_plan(
    *,
    group: ThreadGroup,
    launch: LaunchFacts,
    dtype: Any,
    items_per_thread: int,
    mode: str,
    rank_dtype: Any = None,
    valid_flag_dtype: Any = None,
    warp_time_slicing: bool = False,
    source: str = "cutlass_root",
) -> GroupLoweringPlan:
    """Build the canonical shared-core plan for one CUTLASS group exchange."""

    primitive = make_block_exchange_semantics(
        dtype=dtype,
        items_per_thread=items_per_thread,
        mode=_normalize_exchange_mode(mode),
        value_form="out_of_place",
        warp_time_slicing=warp_time_slicing,
        rank_dtype=rank_dtype,
        valid_flag_dtype=valid_flag_dtype,
    )
    call = make_group_primitive_call(
        group,
        GroupExchangeSemantics(primitive),
        source=source,
    )
    return plan_group_primitive(call, launch)


def _resolve_group_for_exchange(
    group: ThreadGroup,
    launch_kwargs: dict[str, Any],
) -> ThreadGroup:
    return _resolve_collective_group(
        group,
        launch_kwargs,
        feature="exchange",
    )


def _exchange(
    group: ThreadGroup,
    value: Any,
    /,
    *args: Any,
    mode: str = "striped_to_blocked",
    output: ThreadData | None = None,
    ranks: ThreadData | None = None,
    valid_flags: ThreadData | None = None,
    warp_time_slicing: bool = False,
    source: str = "cutlass_root",
    **kwargs: Any,
) -> Any:
    """Internal group exchange entrypoint shared by root and scoped APIs."""

    from ._dsl._launch import infer_launch_facts, pop_launch_metadata
    from ._dsl._scope import merge_payload, validate_no_extra_args

    payload = merge_payload(
        _SCOPE,
        "exchange",
        {
            "group": group,
            "value": value,
            "args": args,
            "mode": mode,
            "output": output,
            "ranks": ranks,
            "valid_flags": valid_flags,
            "warp_time_slicing": warp_time_slicing,
        },
        kwargs,
    )
    launch_kwargs = pop_launch_metadata(kwargs)
    validate_no_extra_args(
        _SCOPE,
        "exchange",
        args=payload.pop("args"),
        kwargs=kwargs,
        expected=(
            "expects a ThreadGroup and one positional ThreadData value, with "
            "an optional mode selector"
        ),
    )
    if not isinstance(group, ThreadGroup):
        raise TypeError(f"{_SCOPE}.exchange group must be a ThreadGroup")
    value = _coerce_thread_payload(
        value,
        scope=_SCOPE,
        primitive_name="exchange",
        arg_name="value",
        common_root_payload_kind="thread_data",
    )
    if not isinstance(value, ThreadData):
        raise TypeError(f"{_SCOPE}.exchange value must be ThreadData")
    if value.items_per_thread > MAX_GROUP_EXCHANGE_ITEMS_PER_THREAD:
        raise NotImplementedError(
            f"{_SCOPE}.exchange supports at most "
            f"{MAX_GROUP_EXCHANGE_ITEMS_PER_THREAD} items per thread"
        )
    if output is not None and not isinstance(output, ThreadData):
        raise TypeError(f"{_SCOPE}.exchange output must be ThreadData")
    if ranks is not None and not isinstance(ranks, ThreadData):
        raise TypeError(f"{_SCOPE}.exchange ranks must be ThreadData")
    if valid_flags is not None and not isinstance(valid_flags, ThreadData):
        raise TypeError(f"{_SCOPE}.exchange valid_flags must be ThreadData")
    if not isinstance(warp_time_slicing, bool):
        raise TypeError(f"{_SCOPE}.exchange warp_time_slicing must be a bool")

    mode = _normalize_exchange_mode(mode)
    _validate_group_for_exchange(group)
    launch = infer_launch_facts(
        launch_kwargs,
        scope=_SCOPE,
        primitive_name="exchange",
    )
    validated_group = _resolve_collective_group_from_launch(
        group,
        launch,
        feature="exchange",
    )
    assert validated_group.hierarchy is not None
    _require_complete_warp_partition(
        validated_group,
        feature="exchange",
        exact_block_dim=validated_group.hierarchy.block_dim,
    )

    from ._dsl import _cub_exchange_provider as _provider

    provider_kwargs = {
        "group": group,
        "launch": launch,
        "value": value,
        "mode": mode,
        "output": output,
        "ranks": ranks,
        "valid_flags": valid_flags,
        "source": source,
    }
    if warp_time_slicing:
        provider_kwargs["warp_time_slicing"] = True
    return _provider.provider_exchange(
        **provider_kwargs,
    )


def exchange(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    mode: str = "striped_to_blocked",
    ranks: ThreadData | None = None,
    valid_flags: ThreadData | None = None,
    warp_time_slicing: bool = False,
) -> Any:
    """Exchange a register payload across a complete block or physical warp.

    ``group`` always names the participating threads explicitly.
    Block groups support every public CUB BlockExchange layout, including the
    warp-striped and scatter variants. Physical and logical warp groups support
    striped/blocked conversion and scatter-to-striped. Scatter modes require a
    matching ``ranks`` payload; flagged block scatter additionally requires
    ``valid_flags``. ``warp_time_slicing`` applies only to block groups.

    The item count comes from ``ThreadData``, an rmem tensor, or ``TensorSSA``.
    Every group member must invoke the collective in converged control flow.
    Warp exchange requires the enclosing CTA to contain complete physical warps.
    """

    return _exchange(
        group,
        value,
        mode=mode,
        ranks=ranks,
        valid_flags=valid_flags,
        warp_time_slicing=warp_time_slicing,
        output=None,
        source="cutlass_root",
    )


__all__ = ["exchange"]
