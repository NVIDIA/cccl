# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from enum import IntEnum, auto
from typing import Any

from cuda.coop._core.block import BlockExchangeMode

from .._scope import BLOCK_SCOPE as _SCOPE
from .._scope import merge_block_payload as merge_payload
from .._scope import validate_no_extra_block_args as validate_no_extra_args
from ._dispatch import dispatch_primitive, register_primitive_impl

_DEFAULT_EXCHANGE_MODE = "striped_to_blocked"


class BlockExchangeType(IntEnum):
    """Select the CUB ``BlockExchange`` mode for ``coop._block.exchange``."""

    StripedToBlocked = auto()
    BlockedToStriped = auto()
    WarpStripedToBlocked = auto()
    BlockedToWarpStriped = auto()
    ScatterToBlocked = auto()
    ScatterToStriped = auto()
    ScatterToStripedGuarded = auto()
    ScatterToStripedFlagged = auto()


_EXCHANGE_MODE_ALIASES = {
    "striped_to_blocked": "striped_to_blocked",
    "striped-to-blocked": "striped_to_blocked",
    "stripedtoblocked": "striped_to_blocked",
    "blocked_to_striped": "blocked_to_striped",
    "blocked-to-striped": "blocked_to_striped",
    "blockedtostriped": "blocked_to_striped",
    "blocked_to_warp_striped": "blocked_to_warp_striped",
    "blocked-to-warp-striped": "blocked_to_warp_striped",
    "blockedtowarpstriped": "blocked_to_warp_striped",
    "warp_striped_to_blocked": "warp_striped_to_blocked",
    "warp-striped-to-blocked": "warp_striped_to_blocked",
    "warpstripedtoblocked": "warp_striped_to_blocked",
    "scatter_to_blocked": "scatter_to_blocked",
    "scatter-to-blocked": "scatter_to_blocked",
    "scattertoblocked": "scatter_to_blocked",
    "scatter_to_striped": "scatter_to_striped",
    "scatter-to-striped": "scatter_to_striped",
    "scattertostriped": "scatter_to_striped",
    "scatter_to_striped_guarded": "scatter_to_striped_guarded",
    "scatter-to-striped-guarded": "scatter_to_striped_guarded",
    "scattertostripedguarded": "scatter_to_striped_guarded",
    "scatter_to_striped_flagged": "scatter_to_striped_flagged",
    "scatter-to-striped-flagged": "scatter_to_striped_flagged",
    "scattertostripedflagged": "scatter_to_striped_flagged",
}


_EXCHANGE_TYPE_TO_MODE = {
    BlockExchangeType.StripedToBlocked: "striped_to_blocked",
    BlockExchangeType.BlockedToStriped: "blocked_to_striped",
    BlockExchangeType.WarpStripedToBlocked: "warp_striped_to_blocked",
    BlockExchangeType.BlockedToWarpStriped: "blocked_to_warp_striped",
    BlockExchangeType.ScatterToBlocked: "scatter_to_blocked",
    BlockExchangeType.ScatterToStriped: "scatter_to_striped",
    BlockExchangeType.ScatterToStripedGuarded: "scatter_to_striped_guarded",
    BlockExchangeType.ScatterToStripedFlagged: "scatter_to_striped_flagged",
}


def _normalize_exchange_mode(mode: Any) -> str:
    try:
        exchange_type = BlockExchangeType(mode)
    except (TypeError, ValueError):
        exchange_type = None
    if exchange_type is not None:
        return _EXCHANGE_TYPE_TO_MODE[exchange_type]

    token = getattr(mode, "name", mode)
    try:
        token = str(token).split(".")[-1]
    except Exception:
        pass
    try:
        return _EXCHANGE_MODE_ALIASES[str(token).lower()]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"{_SCOPE}.exchange mode must be 'striped_to_blocked', "
            "'blocked_to_striped', 'blocked_to_warp_striped', "
            "'warp_striped_to_blocked', 'scatter_to_blocked', "
            "'scatter_to_striped', 'scatter_to_striped_guarded', or "
            "'scatter_to_striped_flagged'"
        ) from exc


def _parse_exchange_args(
    args: tuple[Any, ...],
    *,
    mode: str,
    ranks: Any,
    valid_flags: Any,
    output: Any,
) -> tuple[Any, Any, Any]:
    exchange_mode = BlockExchangeMode(mode)
    if len(args) > 3:
        raise TypeError(
            f"{_SCOPE}.exchange accepts at most output_items, "
            "ranks, and valid_flags positional arguments after value"
        )

    if not exchange_mode.uses_ranks:
        if len(args) > 1:
            raise TypeError(
                f"{_SCOPE}.exchange non-scatter modes accept "
                "at most output_items as a positional argument after value"
            )
        if args:
            if output is not None:
                raise TypeError(f"{_SCOPE}.exchange got duplicate output")
            output = args[0]
        if ranks is not None:
            raise TypeError(f"{_SCOPE}.exchange does not accept ranks for {mode}")
        if valid_flags is not None:
            raise TypeError(f"{_SCOPE}.exchange does not accept valid_flags for {mode}")
        return output, ranks, valid_flags

    if len(args) == 1:
        if ranks is None:
            ranks = args[0]
        else:
            if output is not None:
                raise TypeError(f"{_SCOPE}.exchange got duplicate output")
            output = args[0]
    elif len(args) == 2:
        if exchange_mode.uses_valid_flags and ranks is None and valid_flags is None:
            ranks, valid_flags = args
        else:
            if output is not None:
                raise TypeError(f"{_SCOPE}.exchange got duplicate output")
            if ranks is not None:
                output = args[0]
                if valid_flags is not None:
                    raise TypeError(f"{_SCOPE}.exchange got duplicate valid_flags")
                valid_flags = args[1]
            else:
                output, ranks = args
    elif len(args) == 3:
        if output is not None:
            raise TypeError(f"{_SCOPE}.exchange got duplicate output")
        if ranks is not None:
            raise TypeError(f"{_SCOPE}.exchange got duplicate ranks")
        if valid_flags is not None:
            raise TypeError(f"{_SCOPE}.exchange got duplicate valid_flags")
        output, ranks, valid_flags = args

    if ranks is None:
        raise TypeError(f"{_SCOPE}.exchange requires ranks for scatter modes")
    if exchange_mode.uses_valid_flags and valid_flags is None:
        raise TypeError(
            f"{_SCOPE}.exchange requires valid_flags for scatter_to_striped_flagged"
        )
    if not exchange_mode.uses_valid_flags and valid_flags is not None:
        raise TypeError(
            f"{_SCOPE}.exchange accepts valid_flags only for scatter_to_striped_flagged"
        )
    return output, ranks, valid_flags


def _exchange_provider(
    *,
    value: Any,
    args: tuple[Any, ...] = (),
    output: Any = None,
    mode: str = _DEFAULT_EXCHANGE_MODE,
    ranks: Any = None,
    valid_flags: Any = None,
    **kwargs: Any,
) -> Any:
    validate_no_extra_args(
        "exchange",
        args=args,
        kwargs=kwargs,
        expected="expects normalized exchange payload",
    )

    from ... import _group_exchange as _group_exchange_frontend
    from ..._thread_group import this_block

    if mode in {"striped_to_blocked", "blocked_to_striped"}:
        return _group_exchange_frontend._exchange(
            this_block(),
            value,
            mode=mode,
            output=output,
            source="scoped_block",
            **kwargs,
        )

    from .. import _cub_exchange_provider as _provider
    from .._launch import infer_launch_facts

    launch = infer_launch_facts(
        kwargs,
        scope=_SCOPE,
        primitive_name="exchange",
    )
    group = _group_exchange_frontend._resolve_group_for_exchange(
        this_block(),
        kwargs,
    )
    return _provider.provider_exchange(
        group=group,
        launch=launch,
        value=value,
        output=output,
        mode=mode,
        ranks=ranks,
        valid_flags=valid_flags,
        source="scoped_block_compatibility",
    )


_exchange_provider._supports_native_thread_data = True
_exchange_provider._preserves_launch_metadata = True
_exchange_provider._uses_planned_temp_storage = True
_exchange_provider._supports_deferred_temp_storage = True


def exchange(
    value: Any,
    /,
    *args: Any,
    output: Any = None,
    mode: Any = None,
    ranks: Any = None,
    valid_flags: Any = None,
    block_exchange_type: Any = None,
    **kwargs: Any,
) -> Any:
    """Exchange ``ThreadData`` items across the block using a CUB block layout mode.

    The source and output, when provided, are per-thread ``ThreadData`` objects.
    Supported modes mirror CUB ``BlockExchange`` and its
    ``BlockExchangeType`` selector. Scatter modes consume rank metadata; the
    flagged scatter mode also consumes per-item validity flags. Every rank
    used by a scatter write must be in the block tile. Duplicate write ranks
    and destinations without a writer have unspecified values. Guarded scatter
    uses signed rank -1 to suppress an item; flagged scatter ignores ranks only
    for items whose validity flag is false.

    Positional arguments follow the scoped compatibility call shape. In
    scatter modes, a single positional argument after ``value`` is interpreted as
    ``ranks``; pass ``output=...`` when also supplying an output ``ThreadData``.
    Use three positional arguments as ``output_items, ranks, valid_flags`` for
    flagged scatter when all should be positional.
    """
    if block_exchange_type is None:
        selected_mode = _DEFAULT_EXCHANGE_MODE if mode is None else mode
    else:
        selected_mode = block_exchange_type
        if mode is not None and _normalize_exchange_mode(
            mode
        ) != _normalize_exchange_mode(block_exchange_type):
            raise TypeError(
                f"{_SCOPE}.exchange got conflicting mode and block_exchange_type"
            )
    selected_mode = _normalize_exchange_mode(selected_mode)
    output, ranks, valid_flags = _parse_exchange_args(
        args,
        mode=selected_mode,
        output=output,
        ranks=ranks,
        valid_flags=valid_flags,
    )
    structural_payload = {
        "value": value,
        "args": (),
        "output": output,
        "mode": selected_mode,
    }
    if ranks is not None:
        structural_payload["ranks"] = ranks
    if valid_flags is not None:
        structural_payload["valid_flags"] = valid_flags
    payload = merge_payload("exchange", structural_payload, kwargs)
    return dispatch_primitive("exchange", kwargs=payload)


def exchange_striped_to_blocked(
    value: Any,
    /,
    *args: Any,
    output: Any = None,
    **kwargs: Any,
) -> Any:
    """Exchange per-thread items from striped to blocked logical layout."""
    return exchange(
        value,
        *args,
        output=output,
        mode="striped_to_blocked",
        **kwargs,
    )


def exchange_blocked_to_striped(
    value: Any,
    /,
    *args: Any,
    output: Any = None,
    **kwargs: Any,
) -> Any:
    """Exchange per-thread items from blocked to striped logical layout."""
    return exchange(
        value,
        *args,
        output=output,
        mode="blocked_to_striped",
        **kwargs,
    )


def exchange_blocked_to_warp_striped(
    value: Any,
    /,
    *args: Any,
    output: Any = None,
    **kwargs: Any,
) -> Any:
    """Exchange per-thread items from blocked to warp-striped logical layout."""
    return exchange(
        value,
        *args,
        output=output,
        mode="blocked_to_warp_striped",
        **kwargs,
    )


def exchange_warp_striped_to_blocked(
    value: Any,
    /,
    *args: Any,
    output: Any = None,
    **kwargs: Any,
) -> Any:
    """Exchange per-thread items from warp-striped to blocked logical layout."""
    return exchange(
        value,
        *args,
        output=output,
        mode="warp_striped_to_blocked",
        **kwargs,
    )


def exchange_scatter_to_blocked(
    value: Any,
    ranks: Any,
    /,
    *args: Any,
    output: Any = None,
    **kwargs: Any,
) -> Any:
    """Scatter by in-range unique ranks and return blocked logical layout.

    Duplicate ranks and destinations without a writer have unspecified values.
    """
    return exchange(
        value,
        *args,
        output=output,
        mode="scatter_to_blocked",
        ranks=ranks,
        **kwargs,
    )


def exchange_scatter_to_striped(
    value: Any,
    ranks: Any,
    /,
    *args: Any,
    output: Any = None,
    **kwargs: Any,
) -> Any:
    """Scatter by in-range unique ranks and return striped logical layout.

    Duplicate ranks and destinations without a writer have unspecified values.
    """
    return exchange(
        value,
        *args,
        output=output,
        mode="scatter_to_striped",
        ranks=ranks,
        **kwargs,
    )


def exchange_scatter_to_striped_guarded(
    value: Any,
    ranks: Any,
    /,
    *args: Any,
    output: Any = None,
    **kwargs: Any,
) -> Any:
    """Scatter to striped layout; rank -1 suppresses an item.

    Every other rank must address the tile. Duplicate ranks and destinations
    without a writer have unspecified values.
    """
    return exchange(
        value,
        *args,
        output=output,
        mode="scatter_to_striped_guarded",
        ranks=ranks,
        **kwargs,
    )


def exchange_scatter_to_striped_flagged(
    value: Any,
    ranks: Any,
    valid_flags: Any,
    /,
    *args: Any,
    output: Any = None,
    **kwargs: Any,
) -> Any:
    """Scatter valid items to striped layout.

    Every rank whose flag is true must address the tile; ranks whose flag is
    false are ignored. Duplicate valid ranks and destinations without a writer
    have unspecified values.
    """
    return exchange(
        value,
        *args,
        output=output,
        mode="scatter_to_striped_flagged",
        ranks=ranks,
        valid_flags=valid_flags,
        **kwargs,
    )


register_primitive_impl("exchange", impl=_exchange_provider)
