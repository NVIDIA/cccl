# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from enum import IntEnum, auto
from typing import Any

from .._launch import infer_block_dim as _infer_block_dim
from .._launch import resolve_threads_in_warp as _resolve_threads_in_warp
from .._scope import WARP_SCOPE as _SCOPE
from .._scope import merge_warp_payload as merge_payload
from .._scope import validate_no_extra_warp_args as validate_no_extra_args
from ._dispatch import dispatch_primitive, register_primitive_impl

_DEFAULT_EXCHANGE_MODE = "striped_to_blocked"


class WarpExchangeType(IntEnum):
    """Select the CUB ``WarpExchange`` mode for ``coop._warp.exchange``."""

    StripedToBlocked = auto()
    BlockedToStriped = auto()
    ScatterToStriped = auto()


_EXCHANGE_MODE_ALIASES = {
    "striped_to_blocked": "striped_to_blocked",
    "striped-to-blocked": "striped_to_blocked",
    "stripedtoblocked": "striped_to_blocked",
    "blocked_to_striped": "blocked_to_striped",
    "blocked-to-striped": "blocked_to_striped",
    "blockedtostriped": "blocked_to_striped",
    "scatter_to_striped": "scatter_to_striped",
    "scatter-to-striped": "scatter_to_striped",
    "scattertostriped": "scatter_to_striped",
}


_EXCHANGE_TYPE_TO_MODE = {
    WarpExchangeType.StripedToBlocked: "striped_to_blocked",
    WarpExchangeType.BlockedToStriped: "blocked_to_striped",
    WarpExchangeType.ScatterToStriped: "scatter_to_striped",
}


def _normalize_exchange_mode(mode: Any) -> str:
    try:
        exchange_type = WarpExchangeType(mode)
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
            f"{_SCOPE}.exchange mode must be "
            "'striped_to_blocked', 'blocked_to_striped', or 'scatter_to_striped'"
        ) from exc


def _is_scatter_exchange_type(value: Any) -> bool:
    try:
        return _normalize_exchange_mode(value) == "scatter_to_striped"
    except ValueError:
        return False


def _parse_exchange_args(
    args: tuple[Any, ...],
    *,
    mode: Any,
    ranks: Any,
) -> Any:
    if len(args) > 2:
        raise TypeError(
            f"{_SCOPE}.exchange accepts at most output_items "
            "and ranks as positional arguments after value"
        )
    if not args:
        output_items = None
    elif len(args) == 1 and ranks is None and _is_scatter_exchange_type(mode):
        output_items = None
        ranks = args[0]
    else:
        # The CuTe single-phase API returns ThreadData or fills output=...
        # directly. Keep the compatibility positional output_items slot, but
        # route it through the explicit output kwarg before dispatch.
        output_items = args[0]

    if len(args) == 2:
        if ranks is not None:
            raise TypeError(f"{_SCOPE}.exchange got duplicate ranks")
        ranks = args[1]
    return output_items, ranks


def _exchange_provider(
    *,
    value: Any,
    args: tuple[Any, ...] = (),
    output: Any = None,
    mode: Any = _DEFAULT_EXCHANGE_MODE,
    ranks: Any = None,
    threads_in_warp: int = 32,
    **kwargs: Any,
) -> Any:
    validate_no_extra_args(
        "exchange",
        args=args,
        kwargs=kwargs,
        expected="expects normalized exchange payload",
    )

    threads_in_warp = _resolve_threads_in_warp(
        _SCOPE,
        "exchange",
        threads_in_warp,
    )
    normalized_mode = _normalize_exchange_mode(mode)
    if threads_in_warp == 32 and normalized_mode in {
        "striped_to_blocked",
        "blocked_to_striped",
    }:
        from ... import _group_exchange as _group_exchange_frontend
        from ..._thread_group import this_warp

        return _group_exchange_frontend._exchange(
            this_warp(),
            value,
            mode=normalized_mode,
            output=output,
            source="scoped_warp",
            **kwargs,
        )

    block_dim = _infer_block_dim(
        kwargs,
        scope=_SCOPE,
        primitive_name="exchange",
    )
    block_threads = block_dim[0] * block_dim[1] * block_dim[2]
    if block_threads % threads_in_warp != 0:
        raise ValueError(
            f"{_SCOPE}.exchange requires the exact block thread count to be "
            "a multiple of threads_in_warp"
        )

    from . import _provider

    return _provider.provider_exchange(
        value=value,
        output=output,
        mode=normalized_mode,
        ranks=ranks,
        threads_in_warp=threads_in_warp,
        block_threads=block_threads,
    )


_exchange_provider._supports_native_thread_data = True
_exchange_provider._preserves_launch_metadata = True


def exchange(
    value: Any,
    /,
    *args: Any,
    output: Any = None,
    mode: Any = None,
    ranks: Any = None,
    warp_exchange_type: Any = None,
    threads_in_warp: int = 32,
    **kwargs: Any,
) -> Any:
    """Exchange ``ThreadData`` items across each logical warp with CUB.

    Supported modes mirror CUB ``WarpExchange`` and its
    ``WarpExchangeType`` selector: ``StripedToBlocked``, ``BlockedToStriped``,
    and ``ScatterToStriped``. ``threads_in_warp`` selects a power-of-two
    logical warp size in ``[1, 32]``. Logical-warp and scatter compatibility
    routes support up to 4 items per lane; full physical-warp common modes use
    the canonical group provider without that cap. Every compatibility route
    requires an exact block size divisible by ``threads_in_warp``. Scatter mode
    requires an ``Int32`` rank ``ThreadData`` payload with the same
    ``items_per_thread`` as ``value``. Every rank must be in
    ``[0, threads_in_warp * items_per_thread)``; duplicate ranks and
    destinations without a writer have unspecified values.

    Positional arguments follow the scoped compatibility call shape. In
    scatter mode, a single positional argument after ``value`` is interpreted as
    ``ranks``; pass ``output=...`` when also supplying an output ``ThreadData``.
    Use two positional arguments as ``output_items, ranks`` when both should be
    positional.
    """
    if warp_exchange_type is None:
        selected_mode = _DEFAULT_EXCHANGE_MODE if mode is None else mode
    else:
        selected_mode = warp_exchange_type
        if mode is not None and _normalize_exchange_mode(
            mode
        ) != _normalize_exchange_mode(warp_exchange_type):
            raise TypeError(
                f"{_SCOPE}.exchange got conflicting mode and warp_exchange_type"
            )

    output_items, ranks = _parse_exchange_args(
        args,
        mode=selected_mode,
        ranks=ranks,
    )
    if output_items is not None:
        if output is not None:
            raise TypeError(f"{_SCOPE}.exchange got duplicate output")
        output = output_items
    structural_payload = {
        "value": value,
        "args": (),
        "output": output,
        "mode": _normalize_exchange_mode(selected_mode),
        "threads_in_warp": threads_in_warp,
    }
    if ranks is not None:
        structural_payload["ranks"] = ranks
    payload = merge_payload("exchange", structural_payload, kwargs)
    return dispatch_primitive("exchange", kwargs=payload)


def exchange_striped_to_blocked(
    value: Any,
    /,
    *args: Any,
    output: Any = None,
    threads_in_warp: int = 32,
    **kwargs: Any,
) -> Any:
    """Exchange per-lane items from striped to blocked logical layout."""
    return exchange(
        value,
        *args,
        output=output,
        mode="striped_to_blocked",
        threads_in_warp=threads_in_warp,
        **kwargs,
    )


def exchange_blocked_to_striped(
    value: Any,
    /,
    *args: Any,
    output: Any = None,
    threads_in_warp: int = 32,
    **kwargs: Any,
) -> Any:
    """Exchange per-lane items from blocked to striped logical layout."""
    return exchange(
        value,
        *args,
        output=output,
        mode="blocked_to_striped",
        threads_in_warp=threads_in_warp,
        **kwargs,
    )


def exchange_scatter_to_striped(
    value: Any,
    ranks: Any,
    /,
    *args: Any,
    output: Any = None,
    threads_in_warp: int = 32,
    **kwargs: Any,
) -> Any:
    """Scatter by in-range unique ranks and return striped logical layout.

    Ranks must be ``Int32`` values in the logical-warp tile. Duplicate ranks
    and destinations without a writer have unspecified values.
    """
    return exchange(
        value,
        *args,
        output=output,
        mode="scatter_to_striped",
        ranks=ranks,
        threads_in_warp=threads_in_warp,
        **kwargs,
    )


register_primitive_impl("exchange", impl=_exchange_provider)
