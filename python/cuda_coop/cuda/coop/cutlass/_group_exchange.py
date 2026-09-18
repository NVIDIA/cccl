# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Group-first Exchange entry point for CuTe kernels."""

from enum import Enum

from cuda.coop._core import GroupExchangeMode
from cuda.coop._core.thread_group import ThreadGroup

from ._thread_data import ThreadData, _coerce_thread_payload
from ._thread_group import (
    _require_complete_warp_partition,
    _resolve_primitive_group_from_launch,
)

_SCOPE = "cuda.coop.cutlass"
_BLOCK_MODES = frozenset(mode.value for mode in GroupExchangeMode)
_WARP_MODES = frozenset({"striped_to_blocked", "blocked_to_striped"})


def _normalize_exchange_mode(mode, *, group_kind):
    if not isinstance(mode, str) or isinstance(mode, Enum):
        raise TypeError(f"{_SCOPE}.exchange mode must be a compile-time string")
    token = mode.strip().lower().replace("-", "_")
    allowed = _BLOCK_MODES if group_kind == "block" else _WARP_MODES
    if token not in allowed:
        raise ValueError(
            f"{_SCOPE}.exchange mode for {group_kind} groups must be one of: "
            + ", ".join(sorted(allowed))
        )
    return GroupExchangeMode(token)


def _payload(value, *, name):
    value = _coerce_thread_payload(
        value,
        scope=_SCOPE,
        primitive_name="exchange",
        arg_name=name,
        common_root_payload_kind="thread_data",
    )
    if not isinstance(value, ThreadData):
        raise TypeError(f"{_SCOPE}.exchange {name} must be a fixed-size ThreadData")
    return value


def exchange(
    group,
    value,
    /,
    *,
    mode="striped_to_blocked",
    ranks=None,
    valid_flags=None,
    warp_time_slicing=False,
):
    """Return a fresh payload in the requested layout, preserving all inputs.

    Blocks support blocked/striped and warp-striped layouts, plus ranked
    scatter. Physical and logical warps support blocked/striped layouts.
    Warp groups and block warp-striped layouts require complete 32-lane warps.
    All participating group members must reach the primitive.

    Scatter ranks require a signed integer dtype and the same per-thread
    extent as the values. Valid ranks must be unique across the block and
    within its tile. Guarded scatter skips negative ranks. Flagged scatter
    requires non-boolean integer flags of matching extent and scatters items
    with nonzero flags. Unwritten destinations have undefined values.

    Block warp_time_slicing is a compile-time bool and is unavailable for
    guarded or flagged scatter. Scratch allocation and trailing synchronization
    are automatic. Qualified CuTe register payloads are accepted and outputs
    retain the input dtype, extent, and requested alignment.
    """
    if not isinstance(group, ThreadGroup):
        raise TypeError(f"{_SCOPE}.exchange group must be a ThreadGroup")
    if group.kind not in {"block", "warp", "threads_within_warp"}:
        raise NotImplementedError(f"{_SCOPE}.exchange requires a block or warp group")
    mode = _normalize_exchange_mode(mode, group_kind=group.kind)
    if not isinstance(warp_time_slicing, bool):
        raise TypeError(
            f"{_SCOPE}.exchange warp_time_slicing must be a compile-time bool"
        )
    if warp_time_slicing and group.kind != "block":
        raise ValueError(f"{_SCOPE}.exchange warp_time_slicing applies only to blocks")
    if mode.uses_ranks != (ranks is not None):
        requirement = "requires" if mode.uses_ranks else "does not accept"
        raise ValueError(f"{_SCOPE}.exchange {mode.value} {requirement} ranks")
    if mode.uses_valid_flags != (valid_flags is not None):
        requirement = "requires" if mode.uses_valid_flags else "does not accept"
        raise ValueError(f"{_SCOPE}.exchange {mode.value} {requirement} valid_flags")
    value = _payload(value, name="value")
    ranks = None if ranks is None else _payload(ranks, name="ranks")
    valid_flags = (
        None if valid_flags is None else _payload(valid_flags, name="valid_flags")
    )

    from ._compiler._launch import current_kernel_launch_facts
    from ._lowering import _exchange

    launch = current_kernel_launch_facts()
    group = _resolve_primitive_group_from_launch(group, launch, feature="exchange")
    _require_complete_warp_partition(
        group, feature="exchange", exact_block_dim=launch.exact_block_dim
    )
    value_type, rank_type, flag_type = _exchange._resolve_exchange_operand_types(
        value=value, ranks=ranks, valid_flags=valid_flags
    )
    plan = _exchange._make_group_exchange_plan(
        group=group,
        launch=launch,
        dtype=value_type,
        items_per_thread=value.items_per_thread,
        mode=mode.value,
        rank_dtype=rank_type,
        valid_flag_dtype=flag_type,
        warp_time_slicing=warp_time_slicing,
    ).require_supported()
    return _exchange.provider_exchange(
        plan=plan, value=value, ranks=ranks, valid_flags=valid_flags
    )


__all__ = ["exchange"]
