# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CUTLASS group-first public-CUB shuffle entrypoint."""

from __future__ import annotations

from numbers import Integral
from typing import Any

from cuda.coop._core import (
    ArgumentBinding,
    GroupLoweringPlan,
    GroupShuffleSemantics,
    LaunchFacts,
    make_group_primitive_call,
    plan_group_primitive,
)
from cuda.coop._core.block import BlockShuffleMode, make_block_shuffle_semantics

from ._internal import ThreadData
from ._internal._thread_data import _coerce_thread_payload
from ._thread_group import ThreadGroup, _resolve_collective_group_from_launch

_SCOPE = __name__.rsplit(".", 1)[0]

_MODE_ALIASES = {
    "down": BlockShuffleMode.DOWN,
    "offset": BlockShuffleMode.OFFSET,
    "rotate": BlockShuffleMode.ROTATE,
    "shuffle_down": BlockShuffleMode.DOWN,
    "shuffle_offset": BlockShuffleMode.OFFSET,
    "shuffle_rotate": BlockShuffleMode.ROTATE,
    "shuffle_up": BlockShuffleMode.UP,
    "up": BlockShuffleMode.UP,
}


def _normalize_shuffle_mode(mode: Any) -> BlockShuffleMode:
    try:
        return BlockShuffleMode(mode)
    except (TypeError, ValueError):
        token = getattr(mode, "name", mode)
        token = str(token).split(".")[-1].replace("-", "_").lower()
        try:
            return _MODE_ALIASES[token]
        except KeyError as exc:
            raise ValueError(
                f"{_SCOPE}.shuffle mode must be 'up', 'down', 'offset', or 'rotate'"
            ) from exc


def _normalize_shuffle_route(
    value: Any,
    *,
    mode: BlockShuffleMode,
    distance: Any,
    block_prefix: Any,
    block_suffix: Any,
) -> Any:
    is_thread_data = isinstance(value, ThreadData)
    if isinstance(distance, bool):
        raise TypeError(f"{_SCOPE}.shuffle distance must be an integer, not bool")
    if is_thread_data:
        if not isinstance(distance, Integral):
            raise TypeError(
                f"{_SCOPE}.shuffle ThreadData Up/Down requires the "
                "compile-time unit distance 1; dynamic distance is unsupported"
            )
        distance = int(distance)
        if mode not in {BlockShuffleMode.UP, BlockShuffleMode.DOWN}:
            raise NotImplementedError(
                f"{_SCOPE}.shuffle ThreadData supports only public-CUB Up/Down "
                "unit-shift forms"
            )
        if distance != 1:
            raise NotImplementedError(
                f"{_SCOPE}.shuffle ThreadData Up/Down supports only distance=1"
            )
        if block_prefix is not None and block_suffix is not None:
            raise ValueError(
                f"{_SCOPE}.shuffle accepts only one of block_prefix or block_suffix"
            )
        if block_prefix is not None and mode is not BlockShuffleMode.DOWN:
            raise ValueError(
                f"{_SCOPE}.shuffle block_prefix is valid only for ThreadData Down"
            )
        if block_suffix is not None and mode is not BlockShuffleMode.UP:
            raise ValueError(
                f"{_SCOPE}.shuffle block_suffix is valid only for ThreadData Up"
            )
        return distance

    if mode not in {BlockShuffleMode.OFFSET, BlockShuffleMode.ROTATE}:
        raise NotImplementedError(
            f"{_SCOPE}.shuffle scalar values support only public-CUB Offset/Rotate"
        )
    if block_prefix is not None or block_suffix is not None:
        raise NotImplementedError(
            f"{_SCOPE}.shuffle scalar Offset/Rotate does not return block "
            "prefix or suffix outputs"
        )
    if isinstance(distance, Integral):
        distance = int(distance)
    if mode is BlockShuffleMode.ROTATE and isinstance(distance, int) and distance < 0:
        raise ValueError(f"{_SCOPE}.shuffle Rotate distance must be non-negative")
    return distance


def _make_group_shuffle_plan(
    *,
    group: ThreadGroup,
    launch: LaunchFacts,
    dtype: Any,
    items_per_thread: int | None,
    mode: Any,
    block_prefix: bool = False,
    block_suffix: bool = False,
    source: str = "cutlass_root",
) -> GroupLoweringPlan:
    """Build the canonical CUB BlockShuffle plan."""

    mode = _normalize_shuffle_mode(mode)
    primitive = make_block_shuffle_semantics(
        dtype=dtype,
        mode=mode,
        items_per_thread=items_per_thread,
        distance=(
            ArgumentBinding.runtime()
            if items_per_thread is None
            else ArgumentBinding.omitted()
        ),
        block_prefix=block_prefix,
        block_suffix=block_suffix,
    )
    call = make_group_primitive_call(
        group,
        GroupShuffleSemantics(primitive),
        source=source,
    )
    return plan_group_primitive(call, launch)


def _shuffle(
    group: ThreadGroup,
    value: Any,
    /,
    *args: Any,
    mode: Any = BlockShuffleMode.DOWN,
    distance: Any = 1,
    block_prefix: Any = None,
    block_suffix: Any = None,
    source: str = "cutlass_root",
    **kwargs: Any,
) -> Any:
    """Internal entrypoint shared by root and scoped compatibility APIs."""

    from ._dsl._launch import infer_launch_facts, pop_launch_metadata
    from ._dsl._scope import merge_payload, validate_no_extra_args

    payload = merge_payload(
        _SCOPE,
        "shuffle",
        {"group": group, "value": value, "args": args},
        kwargs,
    )
    launch_kwargs = pop_launch_metadata(kwargs)
    validate_no_extra_args(
        _SCOPE,
        "shuffle",
        args=payload.pop("args"),
        kwargs=kwargs,
        expected=(
            "expects a ThreadGroup and one positional scalar or ThreadData "
            "value, with a public-CUB-compatible mode"
        ),
    )
    if not isinstance(group, ThreadGroup):
        raise TypeError(f"{_SCOPE}.shuffle group must be a ThreadGroup")
    if group.kind != "block":
        raise NotImplementedError(
            f"{_SCOPE}.shuffle currently lowers only this_block groups"
        )
    value = _coerce_thread_payload(
        value,
        scope=_SCOPE,
        primitive_name="shuffle",
        arg_name="value",
        common_root_payload_kind="thread_data",
    )
    mode = _normalize_shuffle_mode(mode)
    distance = _normalize_shuffle_route(
        value,
        mode=mode,
        distance=distance,
        block_prefix=block_prefix,
        block_suffix=block_suffix,
    )
    launch = infer_launch_facts(
        launch_kwargs,
        scope=_SCOPE,
        primitive_name="shuffle",
    )
    validated_group = _resolve_collective_group_from_launch(
        group,
        launch,
        feature="shuffle",
    )

    from ._dsl import _cub_shuffle_provider as _provider

    return _provider.provider_shuffle(
        group=validated_group,
        launch=launch,
        value=value,
        mode=mode,
        distance=distance,
        block_prefix=block_prefix,
        block_suffix=block_suffix,
        source=source,
    )


def shuffle(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    mode: Any = BlockShuffleMode.DOWN,
    distance: Any = 1,
    block_prefix: Any = None,
    block_suffix: Any = None,
) -> Any:
    """Shuffle scalar or fixed-size register values across a CUDA block."""

    return _shuffle(
        group,
        value,
        mode=mode,
        distance=distance,
        block_prefix=block_prefix,
        block_suffix=block_suffix,
        source="cutlass_root",
    )


__all__ = ["shuffle"]
