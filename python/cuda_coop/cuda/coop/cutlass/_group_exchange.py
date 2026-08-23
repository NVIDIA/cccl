# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CUTLASS group-first exchange entrypoint."""

from __future__ import annotations

from typing import Any

from cuda.coop._core import (
    GroupExchangeMode,
)

from ._thread_data import ThreadData, _coerce_thread_payload
from ._thread_group import (
    ThreadGroup,
    _require_complete_warp_partition,
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
    if not isinstance(group, ThreadGroup):
        raise TypeError(f"{_SCOPE}.exchange group must be a ThreadGroup")
    if group.kind not in {"block", "warp", "threads_within_warp"}:
        raise NotImplementedError(
            f"{_SCOPE}.exchange currently lowers CUB exchanges only for "
            "block, physical-warp, and logical-warp groups"
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
) -> ThreadData:
    """Exchange a register payload across a complete block or warp group.

    Block groups support every public CUB ``BlockExchange`` layout, including
    warp-striped and scatter variants. Physical and logical warp groups support
    striped/blocked conversion and scatter-to-striped. Scatter modes require a
    matching ``ranks`` payload; flagged block scatter also requires
    ``valid_flags``. ``warp_time_slicing`` applies only to block groups and is
    unavailable for guarded or flagged scatter-to-striped modes.
    """

    _validate_group_for_exchange(group)
    value = _coerce_thread_payload(
        value,
        scope=_SCOPE,
        primitive_name="exchange",
        arg_name="value",
        common_root_payload_kind="thread_data",
    )
    if not isinstance(value, ThreadData):
        raise TypeError(f"{_SCOPE}.exchange value must be ThreadData")
    if ranks is not None and not isinstance(ranks, ThreadData):
        raise TypeError(f"{_SCOPE}.exchange ranks must be ThreadData")
    if valid_flags is not None and not isinstance(valid_flags, ThreadData):
        raise TypeError(f"{_SCOPE}.exchange valid_flags must be ThreadData")
    if not isinstance(warp_time_slicing, bool):
        raise TypeError(f"{_SCOPE}.exchange warp_time_slicing must be a bool")

    mode = _normalize_exchange_mode(mode)
    from ._compiler._launch import infer_launch_facts

    launch = infer_launch_facts({}, scope=_SCOPE, primitive_name="exchange")
    if not launch.is_verified("exact_block_dim"):
        raise NotImplementedError(
            f"{_SCOPE}.exchange requires exact block dimensions from verified "
            "compiler launch facts"
        )
    resolved_group = _resolve_collective_group_from_launch(
        group,
        launch,
        feature="exchange",
    )
    assert resolved_group.hierarchy is not None
    _require_complete_warp_partition(
        resolved_group,
        feature="exchange",
        exact_block_dim=resolved_group.hierarchy.block_dim,
    )

    from ._lowering import _exchange as _provider

    value_type, rank_type, valid_flag_type = _provider._resolve_exchange_operand_types(
        value=value,
        ranks=ranks,
        valid_flags=valid_flags,
    )
    plan = _provider._make_group_exchange_plan(
        group=resolved_group,
        launch=launch,
        dtype=value_type,
        items_per_thread=value.items_per_thread,
        mode=mode,
        rank_dtype=rank_type,
        valid_flag_dtype=valid_flag_type,
        warp_time_slicing=warp_time_slicing,
    ).require_supported()
    return _provider.provider_exchange(
        plan=plan,
        value=value,
        ranks=ranks,
        valid_flags=valid_flags,
    )


__all__ = ["exchange"]
