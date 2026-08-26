# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CUTLASS group-first discontinuity entrypoint."""

from __future__ import annotations

import operator
from typing import Any

from cuda.coop._core.block import (
    BlockDiscontinuityMode,
)

from ._thread_data import _coerce_thread_payload
from ._thread_group import ThreadGroup, _resolve_collective_group_from_launch

_SCOPE = __name__.rsplit(".", 1)[0]

_MODE_ALIASES = {
    "head": BlockDiscontinuityMode.HEADS,
    "heads": BlockDiscontinuityMode.HEADS,
    "tail": BlockDiscontinuityMode.TAILS,
    "tails": BlockDiscontinuityMode.TAILS,
    "both": BlockDiscontinuityMode.HEADS_AND_TAILS,
    "head_tail": BlockDiscontinuityMode.HEADS_AND_TAILS,
    "heads_and_tails": BlockDiscontinuityMode.HEADS_AND_TAILS,
    "headsandtails": BlockDiscontinuityMode.HEADS_AND_TAILS,
}


def _normalize_discontinuity_mode(mode: Any) -> BlockDiscontinuityMode:
    try:
        return BlockDiscontinuityMode(mode)
    except (TypeError, ValueError):
        token = getattr(mode, "name", mode)
        token = str(token).split(".")[-1].replace("-", "_").lower()
        try:
            return _MODE_ALIASES[token]
        except KeyError as exc:
            raise ValueError(
                f"{_SCOPE}.discontinuity mode must be 'heads', 'tails', or "
                "'heads_and_tails'"
            ) from exc


def _validate_flag_op(flag_op: Any) -> None:
    if (
        flag_op is None
        or flag_op is operator.ne
        or (isinstance(flag_op, str) and flag_op in {"!=", "ne", "not_equal"})
    ):
        return
    module = getattr(flag_op, "__module__", "")
    name = getattr(flag_op, "__name__", "")
    if (module, name) in {("_operator", "ne"), ("operator", "ne")}:
        return
    raise NotImplementedError(
        f"{_SCOPE}.discontinuity currently supports the built-in inequality "
        "operation only"
    )


def _discontinuity(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    mode: Any = BlockDiscontinuityMode.HEADS,
    flag_op: Any = None,
    tile_predecessor_item: Any = None,
    tile_successor_item: Any = None,
    temp_storage: Any = None,
) -> Any:
    """Internal implementation for qualified and common-root calls."""

    from ._compiler._launch import infer_launch_facts

    if not isinstance(group, ThreadGroup):
        raise TypeError(f"{_SCOPE}.discontinuity group must be a ThreadGroup")
    if group.kind != "block":
        raise NotImplementedError(
            f"{_SCOPE}.discontinuity currently lowers only this_block groups"
        )
    value = _coerce_thread_payload(
        value,
        scope=_SCOPE,
        primitive_name="discontinuity",
        arg_name="value",
        common_root_payload_kind="thread_data",
    )
    mode = _normalize_discontinuity_mode(mode)
    _validate_flag_op(flag_op)
    launch = infer_launch_facts({}, scope=_SCOPE, primitive_name="discontinuity")
    validated_group = _resolve_collective_group_from_launch(
        group,
        launch,
        feature="discontinuity",
    )

    from ._lowering import _discontinuity as _provider

    return _provider.provider_discontinuity(
        group=validated_group,
        launch=launch,
        value=value,
        mode=mode,
        tile_predecessor_item=tile_predecessor_item,
        tile_successor_item=tile_successor_item,
        source="cutlass_root",
        temp_storage=temp_storage,
    )


def discontinuity(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    mode: Any = BlockDiscontinuityMode.HEADS,
    tile_predecessor_item: Any = None,
    tile_successor_item: Any = None,
    temp_storage: Any = None,
    flag_op: Any = None,
) -> Any:
    """Flag adjacent-item discontinuities across an explicit CUDA block.

    The input may be a scalar, ``ThreadData``, an rmem tensor, or ``TensorSSA``.
    ``TempStorage`` selects exact caller-owned block scratch, whether its
    capacity is fixed explicitly or inferred after tracing.
    """

    return _discontinuity(
        group,
        value,
        mode=mode,
        flag_op=flag_op,
        tile_predecessor_item=tile_predecessor_item,
        tile_successor_item=tile_successor_item,
        temp_storage=temp_storage,
    )


__all__ = ["discontinuity"]
