"""User-facing API documentation stubs for ``cuda.coop.block``.

This module is used exclusively for documentation generation. The functions
here are non-functional stubs that describe the public call signatures and
behavior without exposing construction helpers or internal two-phase methods.
"""

from __future__ import annotations

from typing import Callable


class BlockExchangeType:
    """Supported exchange patterns.

    - ``StripedToBlocked``: Convert striped layout to blocked layout.
    - ``BlockedToStriped``: Convert blocked layout to striped layout.
    - ``WarpStripedToBlocked``: Convert warp-striped layout to blocked layout.
    - ``BlockedToWarpStriped``: Convert blocked layout to warp-striped layout.
    - ``ScatterToBlocked``: Scatter items to a blocked layout using ranks.
    - ``ScatterToStriped``: Scatter items to a striped layout using ranks.
    - ``ScatterToStripedGuarded``: Scatter to striped layout with bounds checks.
    - ``ScatterToStripedFlagged``: Scatter to striped layout using validity flags.
    """

    StripedToBlocked = "StripedToBlocked"
    BlockedToStriped = "BlockedToStriped"
    WarpStripedToBlocked = "WarpStripedToBlocked"
    BlockedToWarpStriped = "BlockedToWarpStriped"
    ScatterToBlocked = "ScatterToBlocked"
    ScatterToStriped = "ScatterToStriped"
    ScatterToStripedGuarded = "ScatterToStripedGuarded"
    ScatterToStripedFlagged = "ScatterToStripedFlagged"


class BlockDiscontinuityType:
    """Supported discontinuity flagging patterns.

    - ``HEADS``: Flag the start of each run.
    - ``TAILS``: Flag the end of each run.
    - ``HEADS_AND_TAILS``: Return both head and tail flags.
    """

    HEADS = "HEADS"
    TAILS = "TAILS"
    HEADS_AND_TAILS = "HEADS_AND_TAILS"


class BlockAdjacentDifferenceType:
    """Supported adjacent difference patterns.

    - ``SubtractLeft``: Subtract the left neighbor for each item.
    - ``SubtractRight``: Subtract the right neighbor for each item.
    """

    SubtractLeft = "SubtractLeft"
    SubtractRight = "SubtractRight"


class BlockShuffleType:
    """Supported shuffle patterns.

    - ``Offset``: Shift items by a fixed offset.
    - ``Rotate``: Rotate items within the block.
    - ``Up``: Shift items upward by a fixed offset.
    - ``Down``: Shift items downward by a fixed offset.
    """

    Offset = "Offset"
    Rotate = "Rotate"
    Up = "Up"
    Down = "Down"


def exclusive_sum(
    items,
    items_per_thread: int,
    *,
    prefix_op: Callable | None = None,
    algorithm: str | None = None,
    temp_storage=None,
    methods: dict | None = None,
):
    """Exclusive block-wide sum for per-thread items."""


def inclusive_sum(
    items,
    items_per_thread: int,
    *,
    prefix_op: Callable | None = None,
    algorithm: str | None = None,
    temp_storage=None,
    methods: dict | None = None,
):
    """Inclusive block-wide sum for per-thread items."""


def exclusive_scan(
    items,
    items_per_thread: int,
    *,
    scan_op: str | Callable,
    prefix_op: Callable | None = None,
    algorithm: str | None = None,
    temp_storage=None,
    methods: dict | None = None,
):
    """Exclusive block-wide scan with a custom operator."""


def inclusive_scan(
    items,
    items_per_thread: int,
    *,
    scan_op: str | Callable,
    prefix_op: Callable | None = None,
    algorithm: str | None = None,
    temp_storage=None,
    methods: dict | None = None,
):
    """Inclusive block-wide scan with a custom operator."""


def scan(
    items,
    output=None,
    *,
    items_per_thread: int | None = None,
    mode: str | None = None,
    scan_op: str | Callable | None = None,
    prefix_op: Callable | None = None,
    algorithm: str | None = None,
    temp_storage=None,
    methods: dict | None = None,
):
    """Flexible block-wide scan entry point."""


def reduce(
    items,
    items_per_thread: int = 1,
    *,
    binary_op: Callable | None = None,
    algorithm: str | None = None,
    num_valid: int | None = None,
    temp_storage=None,
    methods: dict | None = None,
):
    """Block-wide reduction with a custom operator."""


def sum(
    items,
    items_per_thread: int = 1,
    *,
    algorithm: str | None = None,
    num_valid: int | None = None,
    temp_storage=None,
    methods: dict | None = None,
):
    """Block-wide sum reduction."""


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
    """Block-wide cooperative load into per-thread items."""


def store(
    output_ptr,
    items,
    items_per_thread: int | None = None,
    *,
    algorithm: str | None = None,
    num_valid: int | None = None,
    temp_storage=None,
):
    """Block-wide cooperative store from per-thread items."""


def exchange(
    items,
    *,
    items_per_thread: int,
    block_exchange_type: BlockExchangeType,
    warp_time_slicing: bool = False,
    ranks=None,
    valid_flags=None,
    temp_storage=None,
):
    """Rearrange items across a block using an exchange pattern."""


def shuffle(
    items,
    *,
    items_per_thread: int,
    block_shuffle_type: BlockShuffleType,
    offset: int | None = None,
    indices=None,
    temp_storage=None,
):
    """Shuffle items across a block."""


def discontinuity(
    items,
    *,
    items_per_thread: int,
    block_discontinuity_type: BlockDiscontinuityType,
    predecessor=None,
    successor=None,
    temp_storage=None,
):
    """Flag discontinuities in a blocked arrangement."""


def adjacent_difference(
    items,
    *,
    items_per_thread: int,
    block_adjacent_difference_type: BlockAdjacentDifferenceType,
    predecessor=None,
    successor=None,
    temp_storage=None,
):
    """Compute adjacent differences in a blocked arrangement."""


def merge_sort_keys(
    keys,
    items_per_thread: int,
    compare_op: Callable,
    *,
    valid_items: int | None = None,
    oob_default=None,
    temp_storage=None,
):
    """Block-wide merge sort for keys."""


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
    """Block-wide merge sort for key-value pairs."""


def radix_sort_keys(
    keys,
    items_per_thread: int,
    *,
    begin_bit: int | None = None,
    end_bit: int | None = None,
    temp_storage=None,
):
    """Block-wide radix sort for keys (ascending)."""


def radix_sort_keys_descending(
    keys,
    items_per_thread: int,
    *,
    begin_bit: int | None = None,
    end_bit: int | None = None,
    temp_storage=None,
):
    """Block-wide radix sort for keys (descending)."""


def radix_sort_pairs(
    keys,
    values,
    items_per_thread: int,
    *,
    begin_bit: int | None = None,
    end_bit: int | None = None,
    temp_storage=None,
):
    """Block-wide radix sort for key-value pairs (ascending)."""


def radix_sort_pairs_descending(
    keys,
    values,
    items_per_thread: int,
    *,
    begin_bit: int | None = None,
    end_bit: int | None = None,
    temp_storage=None,
):
    """Block-wide radix sort for key-value pairs (descending)."""


def radix_rank(
    keys,
    items_per_thread: int,
    *,
    begin_bit: int | None = None,
    end_bit: int | None = None,
    descending: bool = False,
    temp_storage=None,
):
    """Block-wide radix rank for keys."""


def run_length(
    items,
    items_per_thread: int,
    *,
    temp_storage=None,
    total_decoded_size=None,
    relative_offsets=None,
    window_offset: int | None = None,
):
    """Block-wide run-length decode for per-thread items."""


def histogram(
    samples,
    histogram_output,
    *,
    bins: int | None = None,
    temp_storage=None,
):
    """Block-wide histogram (single-phase uses ``samples`` + output)."""


__all__ = [
    "BlockAdjacentDifferenceType",
    "BlockDiscontinuityType",
    "BlockExchangeType",
    "BlockShuffleType",
    "adjacent_difference",
    "discontinuity",
    "exchange",
    "exclusive_scan",
    "exclusive_sum",
    "inclusive_scan",
    "inclusive_sum",
    "load",
    "merge_sort_keys",
    "merge_sort_pairs",
    "radix_rank",
    "radix_sort_keys",
    "radix_sort_keys_descending",
    "radix_sort_pairs",
    "radix_sort_pairs_descending",
    "reduce",
    "run_length",
    "scan",
    "shuffle",
    "store",
    "sum",
    "histogram",
]
