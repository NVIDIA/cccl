# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import operator
from enum import IntEnum, auto
from typing import Any

from cuda.coop._core.block import BlockDiscontinuityMode

from .._scope import BLOCK_SCOPE as _SCOPE
from .._scope import merge_block_payload as merge_payload
from .._scope import validate_no_extra_block_args as validate_no_extra_args
from ._dispatch import dispatch_primitive, register_primitive_impl


class BlockDiscontinuityType(IntEnum):
    """Select head, tail, or paired flags for ``coop._block.discontinuity``."""

    HEADS = auto()
    TAILS = auto()
    HEADS_AND_TAILS = auto()


_DISCONTINUITY_TYPE_ALIASES = {
    "heads": BlockDiscontinuityType.HEADS,
    "head": BlockDiscontinuityType.HEADS,
    "tails": BlockDiscontinuityType.TAILS,
    "tail": BlockDiscontinuityType.TAILS,
    "heads_and_tails": BlockDiscontinuityType.HEADS_AND_TAILS,
    "headsandtails": BlockDiscontinuityType.HEADS_AND_TAILS,
    "head_tail": BlockDiscontinuityType.HEADS_AND_TAILS,
    "headtails": BlockDiscontinuityType.HEADS_AND_TAILS,
    "both": BlockDiscontinuityType.HEADS_AND_TAILS,
}


def _normalize_discontinuity_type(value: Any) -> BlockDiscontinuityType:
    try:
        return BlockDiscontinuityType(value)
    except (TypeError, ValueError):
        pass

    token = getattr(value, "name", value)
    token = str(token).split(".")[-1].replace("-", "_").lower()
    try:
        return _DISCONTINUITY_TYPE_ALIASES[token]
    except KeyError as exc:
        raise ValueError(
            "block_discontinuity_type must be a valid BlockDiscontinuityType value"
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
        f"{_SCOPE}.discontinuity currently supports the "
        "built-in not-equal flag operation only. Arbitrary Python flag_op "
        "callables are not lowered yet."
    )


def _parse_discontinuity_args(
    args: tuple[Any, ...],
    *,
    block_discontinuity_type: Any,
) -> Any:
    if len(args) == 0:
        return block_discontinuity_type
    if len(args) == 1:
        if block_discontinuity_type != BlockDiscontinuityType.HEADS:
            raise TypeError(
                f"{_SCOPE}.discontinuity got duplicate block_discontinuity_type"
            )
        return args[0]
    raise TypeError(
        f"{_SCOPE}.discontinuity accepts at most one extra "
        "positional argument for block_discontinuity_type"
    )


def _discontinuity_flag_heads_provider(
    *,
    value: Any,
    args: tuple[Any, ...] = (),
    **kwargs: Any,
) -> Any:
    if args:
        validate_no_extra_args(
            "discontinuity_flag_heads",
            args=args,
            kwargs={},
            expected="expects one positional value",
        )

    from ... import _group_discontinuity as _group_frontend
    from ..._thread_group import this_block

    return _group_frontend._discontinuity(
        this_block(),
        value,
        mode=BlockDiscontinuityMode.HEADS,
        source="scoped_block",
        **kwargs,
    )


_discontinuity_flag_heads_provider._supports_native_thread_data = True
_discontinuity_flag_heads_provider._preserves_launch_metadata = True
_discontinuity_flag_heads_provider._uses_planned_temp_storage = True
_discontinuity_flag_heads_provider._supports_deferred_temp_storage = True


def _discontinuity_flag_tails_provider(
    *,
    value: Any,
    args: tuple[Any, ...] = (),
    **kwargs: Any,
) -> Any:
    if args:
        validate_no_extra_args(
            "discontinuity_flag_tails",
            args=args,
            kwargs={},
            expected="expects one positional value",
        )

    from ... import _group_discontinuity as _group_frontend
    from ..._thread_group import this_block

    return _group_frontend._discontinuity(
        this_block(),
        value,
        mode=BlockDiscontinuityMode.TAILS,
        source="scoped_block",
        **kwargs,
    )


_discontinuity_flag_tails_provider._supports_native_thread_data = True
_discontinuity_flag_tails_provider._preserves_launch_metadata = True
_discontinuity_flag_tails_provider._uses_planned_temp_storage = True
_discontinuity_flag_tails_provider._supports_deferred_temp_storage = True


def _discontinuity_flag_heads_and_tails_provider(
    *,
    value: Any,
    args: tuple[Any, ...] = (),
    **kwargs: Any,
) -> tuple[Any, Any]:
    if args:
        validate_no_extra_args(
            "discontinuity_flag_heads_and_tails",
            args=args,
            kwargs={},
            expected="expects one positional value",
        )

    from ... import _group_discontinuity as _group_frontend
    from ..._thread_group import this_block

    return _group_frontend._discontinuity(
        this_block(),
        value,
        mode=BlockDiscontinuityMode.HEADS_AND_TAILS,
        source="scoped_block",
        **kwargs,
    )


_discontinuity_flag_heads_and_tails_provider._supports_native_thread_data = True
_discontinuity_flag_heads_and_tails_provider._preserves_launch_metadata = True
_discontinuity_flag_heads_and_tails_provider._uses_planned_temp_storage = True
_discontinuity_flag_heads_and_tails_provider._supports_deferred_temp_storage = True


def discontinuity_flag_heads(
    value: Any,
    /,
    *args: Any,
    temp_storage: Any = None,
    **kwargs: Any,
) -> Any:
    """Return a head flag for each item whose predecessor differs in block order."""
    structural_payload = {
        "value": value,
        "args": args,
    }
    if temp_storage is not None:
        structural_payload["temp_storage"] = temp_storage
    payload = merge_payload(
        "discontinuity_flag_heads",
        structural_payload,
        kwargs,
    )
    return dispatch_primitive("discontinuity_flag_heads", kwargs=payload)


def discontinuity_flag_tails(
    value: Any,
    /,
    *args: Any,
    temp_storage: Any = None,
    **kwargs: Any,
) -> Any:
    """Return a tail flag for each item whose successor differs in block order."""
    structural_payload = {
        "value": value,
        "args": args,
    }
    if temp_storage is not None:
        structural_payload["temp_storage"] = temp_storage
    payload = merge_payload(
        "discontinuity_flag_tails",
        structural_payload,
        kwargs,
    )
    return dispatch_primitive("discontinuity_flag_tails", kwargs=payload)


def discontinuity_flag_heads_and_tails(
    value: Any,
    /,
    *args: Any,
    temp_storage: Any = None,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Return block-wide discontinuity head and tail flags as a pair."""
    structural_payload = {
        "value": value,
        "args": args,
    }
    if temp_storage is not None:
        structural_payload["temp_storage"] = temp_storage
    payload = merge_payload(
        "discontinuity_flag_heads_and_tails",
        structural_payload,
        kwargs,
    )
    return dispatch_primitive("discontinuity_flag_heads_and_tails", kwargs=payload)


def discontinuity(
    value: Any,
    /,
    *args: Any,
    block_discontinuity_type: Any = BlockDiscontinuityType.HEADS,
    flag_op: Any = None,
    temp_storage: Any = None,
    **kwargs: Any,
) -> Any:
    """Return block-wide discontinuity flags selected by an explicit mode.

    ``BlockDiscontinuityType.HEADS`` returns head flags, ``TAILS`` returns tail
    flags, and ``HEADS_AND_TAILS`` returns ``(head_flags, tail_flags)``. The
    typed CUTLASS provider lowers the default adjacent-item not-equal predicate
    through one public-CUB LTO-IR shim; arbitrary ``flag_op`` device callables
    remain unsupported without a suitable device-callable ABI.
    """
    selected = _normalize_discontinuity_type(
        _parse_discontinuity_args(
            args,
            block_discontinuity_type=block_discontinuity_type,
        )
    )
    _validate_flag_op(flag_op)
    mode = {
        BlockDiscontinuityType.HEADS: BlockDiscontinuityMode.HEADS,
        BlockDiscontinuityType.TAILS: BlockDiscontinuityMode.TAILS,
        BlockDiscontinuityType.HEADS_AND_TAILS: (
            BlockDiscontinuityMode.HEADS_AND_TAILS
        ),
    }[selected]
    if mode is BlockDiscontinuityMode.HEADS:
        return discontinuity_flag_heads(
            value,
            temp_storage=temp_storage,
            **kwargs,
        )
    if mode is BlockDiscontinuityMode.TAILS:
        return discontinuity_flag_tails(
            value,
            temp_storage=temp_storage,
            **kwargs,
        )
    return discontinuity_flag_heads_and_tails(
        value,
        temp_storage=temp_storage,
        **kwargs,
    )


register_primitive_impl(
    "discontinuity_flag_heads",
    impl=_discontinuity_flag_heads_provider,
)
register_primitive_impl(
    "discontinuity_flag_tails",
    impl=_discontinuity_flag_tails_provider,
)
register_primitive_impl(
    "discontinuity_flag_heads_and_tails",
    impl=_discontinuity_flag_heads_and_tails_provider,
)
