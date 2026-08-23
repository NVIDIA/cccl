# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import operator
from enum import IntEnum, auto
from typing import Any

from cuda.coop._core.block import BlockAdjacentDifferenceDirection

from .._scope import BLOCK_SCOPE as _SCOPE
from .._scope import merge_block_payload as merge_payload
from .._scope import validate_no_extra_block_args as validate_no_extra_args
from ._dispatch import dispatch_primitive, register_primitive_impl


class BlockAdjacentDifferenceType(IntEnum):
    """Select the neighbor direction for ``coop._block.adjacent_difference``."""

    SubtractLeft = auto()
    SubtractRight = auto()


_ADJACENT_DIFFERENCE_TYPE_ALIASES = {
    "subtractleft": BlockAdjacentDifferenceType.SubtractLeft,
    "subtract_left": BlockAdjacentDifferenceType.SubtractLeft,
    "left": BlockAdjacentDifferenceType.SubtractLeft,
    "subtractright": BlockAdjacentDifferenceType.SubtractRight,
    "subtract_right": BlockAdjacentDifferenceType.SubtractRight,
    "right": BlockAdjacentDifferenceType.SubtractRight,
}


def _normalize_adjacent_difference_type(value: Any) -> BlockAdjacentDifferenceType:
    try:
        return BlockAdjacentDifferenceType(value)
    except (TypeError, ValueError):
        pass

    token = getattr(value, "name", value)
    token = str(token).split(".")[-1].replace("-", "_").lower()
    try:
        return _ADJACENT_DIFFERENCE_TYPE_ALIASES[token]
    except KeyError as exc:
        raise ValueError(
            "block_adjacent_difference_type must be a valid "
            "BlockAdjacentDifferenceType value"
        ) from exc


def _validate_difference_op(difference_op: Any) -> None:
    if difference_op is None or difference_op in {"-", "sub", "subtract", operator.sub}:
        return
    module = getattr(difference_op, "__module__", "")
    name = getattr(difference_op, "__name__", "")
    if (module, name) in {("_operator", "sub"), ("operator", "sub")}:
        return
    raise NotImplementedError(
        f"{_SCOPE}.adjacent_difference currently supports "
        "the built-in subtraction operation only. Arbitrary Python difference_op "
        "callables are not lowered yet."
    )


def _parse_adjacent_difference_args(
    args: tuple[Any, ...],
    *,
    block_adjacent_difference_type: Any,
) -> Any:
    if len(args) == 0:
        return block_adjacent_difference_type
    if len(args) == 1:
        if block_adjacent_difference_type != BlockAdjacentDifferenceType.SubtractLeft:
            raise TypeError(
                f"{_SCOPE}.adjacent_difference got duplicate "
                "block_adjacent_difference_type"
            )
        return args[0]
    raise TypeError(
        f"{_SCOPE}.adjacent_difference accepts at most one "
        "extra positional argument for block_adjacent_difference_type"
    )


def _adjacent_difference_subtract_left_provider(
    *,
    value: Any,
    args: tuple[Any, ...] = (),
    **kwargs: Any,
) -> Any:
    if args:
        validate_no_extra_args(
            "adjacent_difference_subtract_left",
            args=args,
            kwargs={},
            expected="expects one positional value",
        )

    from ... import _group_adjacent_difference as _group_frontend
    from ..._thread_group import this_block

    return _group_frontend._adjacent_difference(
        this_block(),
        value,
        direction=BlockAdjacentDifferenceDirection.LEFT,
        source="scoped_block",
        **kwargs,
    )


_adjacent_difference_subtract_left_provider._supports_native_thread_data = True
_adjacent_difference_subtract_left_provider._preserves_launch_metadata = True
_adjacent_difference_subtract_left_provider._uses_planned_temp_storage = True
_adjacent_difference_subtract_left_provider._supports_deferred_temp_storage = True


def _adjacent_difference_subtract_right_provider(
    *,
    value: Any,
    args: tuple[Any, ...] = (),
    **kwargs: Any,
) -> Any:
    if args:
        validate_no_extra_args(
            "adjacent_difference_subtract_right",
            args=args,
            kwargs={},
            expected="expects one positional value",
        )

    from ... import _group_adjacent_difference as _group_frontend
    from ..._thread_group import this_block

    return _group_frontend._adjacent_difference(
        this_block(),
        value,
        direction=BlockAdjacentDifferenceDirection.RIGHT,
        source="scoped_block",
        **kwargs,
    )


_adjacent_difference_subtract_right_provider._supports_native_thread_data = True
_adjacent_difference_subtract_right_provider._preserves_launch_metadata = True
_adjacent_difference_subtract_right_provider._uses_planned_temp_storage = True
_adjacent_difference_subtract_right_provider._supports_deferred_temp_storage = True


def adjacent_difference_subtract_left(
    value: Any,
    /,
    *args: Any,
    temp_storage: Any = None,
    **kwargs: Any,
) -> Any:
    """Return ``value - predecessor`` for each block-ordered item.

    ``valid_items`` selects CUB's partial-tile overload and
    ``tile_predecessor_item`` supplies an external predecessor.
    """
    structural_payload = {
        "value": value,
        "args": args,
    }
    if temp_storage is not None:
        structural_payload["temp_storage"] = temp_storage
    payload = merge_payload(
        "adjacent_difference_subtract_left",
        structural_payload,
        kwargs,
    )
    return dispatch_primitive("adjacent_difference_subtract_left", kwargs=payload)


def adjacent_difference_subtract_right(
    value: Any,
    /,
    *args: Any,
    temp_storage: Any = None,
    **kwargs: Any,
) -> Any:
    """Return ``value - successor`` for each block-ordered item.

    ``valid_items`` selects CUB's partial-tile overload. A full tile may supply
    an external ``tile_successor_item``.
    """
    structural_payload = {
        "value": value,
        "args": args,
    }
    if temp_storage is not None:
        structural_payload["temp_storage"] = temp_storage
    payload = merge_payload(
        "adjacent_difference_subtract_right",
        structural_payload,
        kwargs,
    )
    return dispatch_primitive("adjacent_difference_subtract_right", kwargs=payload)


def adjacent_difference(
    value: Any,
    /,
    *args: Any,
    block_adjacent_difference_type: Any = BlockAdjacentDifferenceType.SubtractLeft,
    difference_op: Any = None,
    temp_storage: Any = None,
    **kwargs: Any,
) -> Any:
    """Return block-wide adjacent differences selected by an explicit mode.

    ``BlockAdjacentDifferenceType.SubtractLeft`` returns
    ``value - predecessor`` for each block-ordered item. ``SubtractRight``
    returns ``value - successor``. The current CuTe provider lowers the default
    subtraction operation through CUB-backed LTO-IR shims; arbitrary
    ``difference_op`` device callables are rejected until CuTe exposes a
    suitable callable ABI.
    """
    selected = _normalize_adjacent_difference_type(
        _parse_adjacent_difference_args(
            args,
            block_adjacent_difference_type=block_adjacent_difference_type,
        )
    )
    _validate_difference_op(difference_op)
    direction = (
        BlockAdjacentDifferenceDirection.LEFT
        if selected == BlockAdjacentDifferenceType.SubtractLeft
        else BlockAdjacentDifferenceDirection.RIGHT
    )
    if direction is BlockAdjacentDifferenceDirection.LEFT:
        return adjacent_difference_subtract_left(
            value,
            temp_storage=temp_storage,
            **kwargs,
        )
    return adjacent_difference_subtract_right(
        value,
        temp_storage=temp_storage,
        **kwargs,
    )


register_primitive_impl(
    "adjacent_difference_subtract_left",
    impl=_adjacent_difference_subtract_left_provider,
)
register_primitive_impl(
    "adjacent_difference_subtract_right",
    impl=_adjacent_difference_subtract_right_provider,
)
