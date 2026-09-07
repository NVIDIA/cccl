# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from enum import IntEnum, auto
from typing import Any

from cuda.coop._core.block import BlockShuffleMode

from .._scope import BLOCK_SCOPE as _SCOPE
from .._scope import merge_block_payload as merge_payload
from .._scope import validate_no_extra_block_args as validate_no_extra_args
from ._dispatch import dispatch_primitive, register_primitive_impl

_DEFAULT_SHUFFLE_MODE = "down"


class BlockShuffleType(IntEnum):
    """Select the CUB ``BlockShuffle`` mode for ``coop._block.shuffle``."""

    Offset = auto()
    Rotate = auto()
    Up = auto()
    Down = auto()


_SHUFFLE_MODE_ALIASES = {
    "down": "down",
    "offset": "offset",
    "shuffle_down": "down",
    "shuffle_offset": "offset",
    "up": "up",
    "shuffle_up": "up",
    "rotate": "rotate",
    "shuffle_rotate": "rotate",
}


def _normalize_shuffle_mode(mode: Any) -> str:
    try:
        shuffle_type = BlockShuffleType(mode)
    except (TypeError, ValueError):
        shuffle_type = None
    if shuffle_type is not None:
        return BlockShuffleMode.from_cub_method_name(shuffle_type.name).value

    token = getattr(mode, "name", mode)
    token = str(token).split(".")[-1].lower().replace("-", "_")
    try:
        return _SHUFFLE_MODE_ALIASES[token]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"{_SCOPE}.shuffle mode must be 'up', 'down', 'offset', or 'rotate'"
        ) from exc


def _shuffle_provider(
    *,
    value: Any,
    args: tuple[Any, ...] = (),
    mode: str = _DEFAULT_SHUFFLE_MODE,
    distance: int = 1,
    block_prefix: Any = None,
    block_suffix: Any = None,
    **kwargs: Any,
) -> Any:
    if args:
        validate_no_extra_args(
            "shuffle",
            args=args,
            kwargs={},
            expected="expects one positional value",
        )

    from ... import _group_shuffle as _group_frontend
    from ..._thread_group import this_block

    return _group_frontend._shuffle(
        this_block(),
        value,
        mode=mode,
        distance=distance,
        block_prefix=block_prefix,
        block_suffix=block_suffix,
        source="scoped_block",
        **kwargs,
    )


_shuffle_provider._supports_native_thread_data = True
_shuffle_provider._preserves_launch_metadata = True
_shuffle_provider._uses_planned_temp_storage = True


def shuffle(
    value: Any,
    /,
    *args: Any,
    mode: Any = None,
    block_shuffle_type: Any = None,
    distance: int = 1,
    block_prefix: Any = None,
    block_suffix: Any = None,
    **kwargs: Any,
) -> Any:
    """Return a block-wide scalar or ``ThreadData`` shuffle result.

    ``mode`` or the compatibility ``block_shuffle_type`` selector chooses up,
    down, offset, or rotate semantics. Multi-item ``ThreadData`` shuffles can
    optionally report block prefix/suffix values through ``block_prefix`` and
    ``block_suffix`` outputs.
    """
    if block_shuffle_type is None:
        selected_mode = _DEFAULT_SHUFFLE_MODE if mode is None else mode
    else:
        selected_mode = block_shuffle_type
        if mode is not None and _normalize_shuffle_mode(
            mode
        ) != _normalize_shuffle_mode(block_shuffle_type):
            raise TypeError(
                f"{_SCOPE}.shuffle got conflicting mode and block_shuffle_type"
            )
    resolved_mode = _normalize_shuffle_mode(selected_mode)
    structural_payload = {
        "value": value,
        "args": args,
        "mode": resolved_mode,
        "distance": distance,
    }
    if block_prefix is not None:
        structural_payload["block_prefix"] = block_prefix
    if block_suffix is not None:
        structural_payload["block_suffix"] = block_suffix
    payload = merge_payload(
        "shuffle",
        structural_payload,
        kwargs,
    )
    return dispatch_primitive("shuffle", kwargs=payload)


def shuffle_up(
    value: Any,
    /,
    *args: Any,
    distance: int = 1,
    **kwargs: Any,
) -> Any:
    """Return ``shuffle(..., mode="up")`` for block-scoped call sites."""
    return shuffle(value, *args, mode="up", distance=distance, **kwargs)


def shuffle_down(
    value: Any,
    /,
    *args: Any,
    distance: int = 1,
    **kwargs: Any,
) -> Any:
    """Return ``shuffle(..., mode="down")`` for block-scoped call sites."""
    return shuffle(value, *args, mode="down", distance=distance, **kwargs)


def shuffle_offset(
    value: Any,
    /,
    *args: Any,
    distance: int = 1,
    **kwargs: Any,
) -> Any:
    """Return ``shuffle(..., mode="offset")`` for block-scoped call sites."""
    return shuffle(value, *args, mode="offset", distance=distance, **kwargs)


def shuffle_rotate(
    value: Any,
    /,
    *args: Any,
    distance: int = 1,
    **kwargs: Any,
) -> Any:
    """Return ``shuffle(..., mode="rotate")`` for block-scoped call sites."""
    return shuffle(value, *args, mode="rotate", distance=distance, **kwargs)


register_primitive_impl("shuffle", impl=_shuffle_provider)
