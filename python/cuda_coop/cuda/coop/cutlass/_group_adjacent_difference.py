# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CUTLASS group-first adjacent-difference entrypoint."""

from __future__ import annotations

import operator
from numbers import Integral
from typing import Any

from cuda.coop._core import (
    CxxOperator,
    Dependency,
    GroupAdjacentDifferenceSemantics,
    GroupLoweringPlan,
    LaunchFacts,
    make_group_primitive_call,
    plan_group_primitive,
)
from cuda.coop._core.block import (
    BlockAdjacentDifferenceDirection,
    make_block_adjacent_difference_semantics,
)

from ._internal._thread_data import _coerce_thread_payload
from ._thread_group import ThreadGroup, _resolve_collective_group_from_launch

_SCOPE = __name__.rsplit(".", 1)[0]

_DIRECTION_ALIASES = {
    "subtractleft": BlockAdjacentDifferenceDirection.LEFT,
    "subtract_left": BlockAdjacentDifferenceDirection.LEFT,
    "left": BlockAdjacentDifferenceDirection.LEFT,
    "subtractright": BlockAdjacentDifferenceDirection.RIGHT,
    "subtract_right": BlockAdjacentDifferenceDirection.RIGHT,
    "right": BlockAdjacentDifferenceDirection.RIGHT,
}


def _normalize_adjacent_difference_direction(
    direction: Any,
) -> BlockAdjacentDifferenceDirection:
    try:
        return BlockAdjacentDifferenceDirection(direction)
    except (TypeError, ValueError):
        token = getattr(direction, "name", direction)
        token = str(token).split(".")[-1].replace("-", "_").lower()
        try:
            return _DIRECTION_ALIASES[token]
        except KeyError as exc:
            raise ValueError(
                f"{_SCOPE}.adjacent_difference direction must be 'left' or 'right'"
            ) from exc


def _validate_difference_op(difference_op: Any) -> None:
    if difference_op is None or difference_op is operator.sub:
        return
    if isinstance(difference_op, str) and difference_op in {
        "-",
        "sub",
        "subtract",
    }:
        return
    module = getattr(difference_op, "__module__", "")
    name = getattr(difference_op, "__name__", "")
    if (module, name) in {("_operator", "sub"), ("operator", "sub")}:
        return
    raise NotImplementedError(
        f"{_SCOPE}.adjacent_difference currently supports the built-in "
        "subtraction operation only"
    )


def _validate_valid_items(valid_items: Any, *, tile_size: int) -> None:
    if valid_items is None:
        return
    from ._group_reduce import _is_boolean_control

    if _is_boolean_control(valid_items):
        raise TypeError(f"{_SCOPE}.adjacent_difference valid_items must be an integer")
    if isinstance(valid_items, Integral):
        if not 0 <= int(valid_items) <= tile_size:
            raise ValueError(
                f"{_SCOPE}.adjacent_difference static valid_items must be "
                f"between zero and the block tile size ({tile_size})"
            )
        return
    from cutlass.base_dsl.typing import Integer

    if not isinstance(valid_items, Integer):
        raise TypeError(f"{_SCOPE}.adjacent_difference valid_items must be an integer")


def _make_group_adjacent_difference_plan(
    *,
    group: ThreadGroup,
    launch: LaunchFacts,
    dtype: Any,
    items_per_thread: int,
    direction: Any,
    valid_items: Any = None,
    tile_predecessor_item: Any = None,
    tile_successor_item: Any = None,
    source: str = "cutlass_root",
) -> GroupLoweringPlan:
    """Build the canonical shared-core plan for BlockAdjacentDifference."""

    primitive = make_block_adjacent_difference_semantics(
        dtype=dtype,
        items_per_thread=items_per_thread,
        direction=_normalize_adjacent_difference_direction(direction),
        difference_operator=CxxOperator(
            "::cuda::std::minus<T>",
            Dependency("T"),
            name="difference_op",
        ),
        valid_items=valid_items,
        tile_predecessor_item=tile_predecessor_item,
        tile_successor_item=tile_successor_item,
    )
    call = make_group_primitive_call(
        group,
        GroupAdjacentDifferenceSemantics(primitive),
        source=source,
    )
    return plan_group_primitive(call, launch)


def _adjacent_difference(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    direction: Any = BlockAdjacentDifferenceDirection.LEFT,
    difference_op: Any = None,
    valid_items: Any = None,
    tile_predecessor_item: Any = None,
    tile_successor_item: Any = None,
    temp_storage: Any = None,
) -> Any:
    """Internal implementation for qualified and common-root calls."""

    from ._dsl._launch import infer_launch_facts

    if not isinstance(group, ThreadGroup):
        raise TypeError(f"{_SCOPE}.adjacent_difference group must be a ThreadGroup")
    if group.kind != "block":
        raise NotImplementedError(
            f"{_SCOPE}.adjacent_difference currently lowers only this_block groups"
        )
    value = _coerce_thread_payload(
        value,
        scope=_SCOPE,
        primitive_name="adjacent_difference",
        arg_name="value",
        common_root_payload_kind="thread_data",
    )
    direction = _normalize_adjacent_difference_direction(direction)
    _validate_difference_op(difference_op)
    launch = infer_launch_facts({}, scope=_SCOPE, primitive_name="adjacent_difference")
    validated_group = _resolve_collective_group_from_launch(
        group,
        launch,
        feature="adjacent_difference",
    )
    assert validated_group.static_size is not None
    _validate_valid_items(
        valid_items,
        tile_size=validated_group.static_size * getattr(value, "items_per_thread", 1),
    )

    from ._dsl import _cub_adjacent_difference_provider as _provider

    return _provider.provider_adjacent_difference(
        group=validated_group,
        launch=launch,
        value=value,
        direction=direction,
        valid_items=valid_items,
        tile_predecessor_item=tile_predecessor_item,
        tile_successor_item=tile_successor_item,
        source="cutlass_root",
        temp_storage=temp_storage,
    )


def adjacent_difference(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    direction: Any = BlockAdjacentDifferenceDirection.LEFT,
    valid_items: Any = None,
    tile_predecessor_item: Any = None,
    tile_successor_item: Any = None,
    temp_storage: Any = None,
    difference_op: Any = None,
) -> Any:
    """Compute adjacent differences across an explicit CUDA block group.

    Scalar values, ``ThreadData``, rmem tensors, and ``TensorSSA`` are accepted.
    ``direction`` selects subtraction from the left or right neighbor.
    ``valid_items`` selects the public CUB partial-tile overload. A predecessor
    is valid only for left subtraction; a successor is valid only for full-tile
    right subtraction. The built-in subtraction operator is the only supported
    operator. ``TempStorage`` selects exact caller-owned block scratch, whether
    its capacity is fixed explicitly or inferred after tracing.
    """

    return _adjacent_difference(
        group,
        value,
        direction=direction,
        difference_op=difference_op,
        valid_items=valid_items,
        tile_predecessor_item=tile_predecessor_item,
        tile_successor_item=tile_successor_item,
        temp_storage=temp_storage,
    )


__all__ = ["adjacent_difference"]
