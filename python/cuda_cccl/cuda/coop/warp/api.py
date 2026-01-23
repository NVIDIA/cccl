"""User-facing API documentation stubs for ``cuda.coop.warp``.

This module is used exclusively for documentation generation. The functions
here are non-functional stubs that describe the public call signatures and
behavior without exposing construction helpers or internal two-phase methods.
"""

from __future__ import annotations

from typing import Callable


class WarpExchangeType:
    """Supported warp exchange patterns.

    - ``StripedToBlocked``: Convert striped layout to blocked layout.
    - ``BlockedToStriped``: Convert blocked layout to striped layout.
    - ``ScatterToStriped``: Scatter items to a striped layout using ranks.
    """

    StripedToBlocked = "StripedToBlocked"
    BlockedToStriped = "BlockedToStriped"
    ScatterToStriped = "ScatterToStriped"

    StripedToBlocked = "StripedToBlocked"
    BlockedToStriped = "BlockedToStriped"
    ScatterToBlocked = "ScatterToBlocked"
    ScatterToStriped = "ScatterToStriped"
    ScatterToStripedGuarded = "ScatterToStripedGuarded"
    ScatterToStripedFlagged = "ScatterToStripedFlagged"


def exclusive_sum(
    items,
    items_per_thread: int,
    *,
    prefix_op: Callable | None = None,
    algorithm: str | None = None,
    temp_storage=None,
):
    """Exclusive warp-wide sum for per-thread items."""


def inclusive_sum(
    items,
    items_per_thread: int,
    *,
    prefix_op: Callable | None = None,
    algorithm: str | None = None,
    temp_storage=None,
):
    """Inclusive warp-wide sum for per-thread items."""


def exclusive_scan(
    items,
    items_per_thread: int,
    *,
    scan_op: str | Callable,
    prefix_op: Callable | None = None,
    algorithm: str | None = None,
    temp_storage=None,
):
    """Exclusive warp-wide scan with a custom operator."""


def inclusive_scan(
    items,
    items_per_thread: int,
    *,
    scan_op: str | Callable,
    prefix_op: Callable | None = None,
    algorithm: str | None = None,
    temp_storage=None,
):
    """Inclusive warp-wide scan with a custom operator."""


def reduce(
    items,
    items_per_thread: int = 1,
    *,
    binary_op: Callable | None = None,
    algorithm: str | None = None,
    temp_storage=None,
):
    """Warp-wide reduction with a custom operator."""


def sum(
    items,
    items_per_thread: int = 1,
    *,
    algorithm: str | None = None,
    temp_storage=None,
):
    """Warp-wide sum reduction."""


def load(
    input_ptr,
    output,
    items_per_thread: int | None = None,
    *,
    algorithm: str | None = None,
    num_valid: int | None = None,
    oob_default=None,
    temp_storage=None,
):
    """Warp-wide cooperative load into per-thread items."""


def store(
    output_ptr,
    items,
    items_per_thread: int | None = None,
    *,
    algorithm: str | None = None,
    num_valid: int | None = None,
    temp_storage=None,
):
    """Warp-wide cooperative store from per-thread items."""


def exchange(
    items,
    *,
    items_per_thread: int,
    warp_exchange_type: WarpExchangeType,
    ranks=None,
    valid_flags=None,
    temp_storage=None,
):
    """Rearrange items across a warp using an exchange pattern."""


def merge_sort_keys(
    keys,
    items_per_thread: int,
    compare_op: Callable,
    *,
    valid_items: int | None = None,
    oob_default=None,
    temp_storage=None,
):
    """Warp-wide merge sort for keys."""


def merge_sort_pairs(
    keys,
    values,
    items_per_thread: int,
    compare_op: Callable,
    *,
    valid_items: int | None = None,
    oob_default=None,
    temp_storage=None,
):
    """Warp-wide merge sort for key-value pairs."""


__all__ = [
    "WarpExchangeType",
    "exclusive_scan",
    "exclusive_sum",
    "inclusive_scan",
    "inclusive_sum",
    "reduce",
    "sum",
    "load",
    "store",
    "exchange",
    "merge_sort_keys",
    "merge_sort_pairs",
]
