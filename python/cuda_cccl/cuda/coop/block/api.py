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
    """Exclusive block-wide sum for per-thread items.

    Example:
        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_scan_api.py
            :language: python
            :dedent:
            :start-after: example-begin imports
            :end-before: example-end imports

        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_scan_api.py
            :language: python
            :dedent:
            :start-after: example-begin exclusive-sum
            :end-before: example-end exclusive-sum

        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_scan_api.py
            :language: python
            :dedent:
            :start-after: example-begin exclusive-sum-single-input-per-thread
            :end-before: example-end exclusive-sum-single-input-per-thread

        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_scan_api.py
            :language: python
            :dedent:
            :start-after: example-begin exclusive-sum-block-aggregate
            :end-before: example-end exclusive-sum-block-aggregate
    """


def inclusive_sum(
    items,
    items_per_thread: int,
    *,
    prefix_op: Callable | None = None,
    algorithm: str | None = None,
    temp_storage=None,
    methods: dict | None = None,
):
    """Inclusive block-wide sum for per-thread items.

    Example:
        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_scan_api_extra.py
            :language: python
            :dedent:
            :start-after: example-begin inclusive-sum
            :end-before: example-end inclusive-sum
    """


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
    """Exclusive block-wide scan with a custom operator.

    Example:
        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_scan_api_extra.py
            :language: python
            :dedent:
            :start-after: example-begin exclusive-scan
            :end-before: example-end exclusive-scan
    """


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
    """Inclusive block-wide scan with a custom operator.

    Example:
        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_scan_api_extra.py
            :language: python
            :dedent:
            :start-after: example-begin inclusive-scan
            :end-before: example-end inclusive-scan
    """


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
    """Flexible block-wide scan entry point.

    Example:
        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_scan_api.py
            :language: python
            :dedent:
            :start-after: example-begin exclusive-sum
            :end-before: example-end exclusive-sum
    """


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
    """Block-wide reduction with a custom operator.

    Example:
        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_reduce_api.py
            :language: python
            :dedent:
            :start-after: example-begin reduce
            :end-before: example-end reduce

        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_reduce_api.py
            :language: python
            :dedent:
            :start-after: example-begin reduce-temp-storage
            :end-before: example-end reduce-temp-storage

        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_reduce_api.py
            :language: python
            :dedent:
            :start-after: example-begin reduce-two-phase-temp-storage
            :end-before: example-end reduce-two-phase-temp-storage
    """


def sum(
    items,
    items_per_thread: int = 1,
    *,
    algorithm: str | None = None,
    num_valid: int | None = None,
    temp_storage=None,
    methods: dict | None = None,
):
    """Block-wide sum reduction.

    Example:
        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_reduce_api.py
            :language: python
            :dedent:
            :start-after: example-begin sum
            :end-before: example-end sum

        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_reduce_api.py
            :language: python
            :dedent:
            :start-after: example-begin sum-temp-storage
            :end-before: example-end sum-temp-storage

        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_reduce_api.py
            :language: python
            :dedent:
            :start-after: example-begin sum-two-phase-temp-storage
            :end-before: example-end sum-two-phase-temp-storage
    """


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
    """Block-wide cooperative load into per-thread items.

    Example:
        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_load_store_api_single_phase.py
            :language: python
            :dedent:
            :start-after: example-begin imports
            :end-before: example-end imports

        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_load_store_api_single_phase.py
            :language: python
            :dedent:
            :start-after: example-begin load-store-single-phase
            :end-before: example-end load-store-single-phase

        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_load_store_api_single_phase.py
            :language: python
            :dedent:
            :start-after: example-begin load-single-phase-oob-default
            :end-before: example-end load-single-phase-oob-default
    """


def store(
    output_ptr,
    items,
    items_per_thread: int | None = None,
    *,
    algorithm: str | None = None,
    num_valid: int | None = None,
    temp_storage=None,
):
    """Block-wide cooperative store from per-thread items.

    Example:
        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_load_store_api_single_phase.py
            :language: python
            :dedent:
            :start-after: example-begin load-store-single-phase
            :end-before: example-end load-store-single-phase
    """


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
    """Rearrange items across a block using an exchange pattern.

    Example:
        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_exchange_api.py
            :language: python
            :dedent:
            :start-after: example-begin striped-to-blocked
            :end-before: example-end striped-to-blocked

        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_exchange_api.py
            :language: python
            :dedent:
            :start-after: example-begin blocked-to-striped
            :end-before: example-end blocked-to-striped
    """


def shuffle(
    items,
    *,
    items_per_thread: int,
    block_shuffle_type: BlockShuffleType,
    offset: int | None = None,
    indices=None,
    temp_storage=None,
):
    """Shuffle items across a block.

    Example:
        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_shuffle_api.py
            :language: python
            :dedent:
            :start-after: example-begin offset-scalar
            :end-before: example-end offset-scalar

        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_shuffle_api.py
            :language: python
            :dedent:
            :start-after: example-begin rotate-scalar
            :end-before: example-end rotate-scalar

        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_shuffle_api.py
            :language: python
            :dedent:
            :start-after: example-begin up-scalar
            :end-before: example-end up-scalar

        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_shuffle_api.py
            :language: python
            :dedent:
            :start-after: example-begin down-scalar
            :end-before: example-end down-scalar
    """


def discontinuity(
    items,
    *,
    items_per_thread: int,
    block_discontinuity_type: BlockDiscontinuityType,
    predecessor=None,
    successor=None,
    temp_storage=None,
):
    """Flag discontinuities in a blocked arrangement.

    Example:
        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_discontinuity_api.py
            :language: python
            :dedent:
            :start-after: example-begin flag-heads
            :end-before: example-end flag-heads

        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_discontinuity_api.py
            :language: python
            :dedent:
            :start-after: example-begin flag-tails
            :end-before: example-end flag-tails

        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_discontinuity_api.py
            :language: python
            :dedent:
            :start-after: example-begin flag-heads-and-tails
            :end-before: example-end flag-heads-and-tails
    """


def adjacent_difference(
    items,
    *,
    items_per_thread: int,
    block_adjacent_difference_type: BlockAdjacentDifferenceType,
    predecessor=None,
    successor=None,
    temp_storage=None,
):
    """Compute adjacent differences in a blocked arrangement.

    Example:
        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_adjacent_difference_api.py
            :language: python
            :dedent:
            :start-after: example-begin subtract-left
            :end-before: example-end subtract-left

        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_adjacent_difference_api.py
            :language: python
            :dedent:
            :start-after: example-begin subtract-right
            :end-before: example-end subtract-right
    """


def merge_sort_keys(
    keys,
    items_per_thread: int,
    compare_op: Callable,
    *,
    valid_items: int | None = None,
    oob_default=None,
    temp_storage=None,
):
    """Block-wide merge sort for keys.

    Example:
        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_merge_sort_api.py
            :language: python
            :dedent:
            :start-after: example-begin merge-sort
            :end-before: example-end merge-sort
    """


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
    """Block-wide merge sort for key-value pairs.

    Example:
        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_merge_sort_pairs_api.py
            :language: python
            :dedent:
            :start-after: example-begin merge-sort-pairs
            :end-before: example-end merge-sort-pairs
    """


def radix_sort_keys(
    keys,
    items_per_thread: int,
    *,
    begin_bit: int | None = None,
    end_bit: int | None = None,
    temp_storage=None,
):
    """Block-wide radix sort for keys (ascending).

    Example:
        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_radix_sort_api.py
            :language: python
            :dedent:
            :start-after: example-begin radix-sort
            :end-before: example-end radix-sort
    """


def radix_sort_keys_descending(
    keys,
    items_per_thread: int,
    *,
    begin_bit: int | None = None,
    end_bit: int | None = None,
    temp_storage=None,
):
    """Block-wide radix sort for keys (descending).

    Example:
        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_radix_sort_api.py
            :language: python
            :dedent:
            :start-after: example-begin radix-sort-descending
            :end-before: example-end radix-sort-descending
    """


def radix_sort_pairs(
    keys,
    values,
    items_per_thread: int,
    *,
    begin_bit: int | None = None,
    end_bit: int | None = None,
    temp_storage=None,
):
    """Block-wide radix sort for key-value pairs (ascending).

    Example:
        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_radix_sort_pairs_api.py
            :language: python
            :dedent:
            :start-after: example-begin radix-sort-pairs
            :end-before: example-end radix-sort-pairs
    """


def radix_sort_pairs_descending(
    keys,
    values,
    items_per_thread: int,
    *,
    begin_bit: int | None = None,
    end_bit: int | None = None,
    temp_storage=None,
):
    """Block-wide radix sort for key-value pairs (descending).

    Example:
        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_radix_sort_pairs_api.py
            :language: python
            :dedent:
            :start-after: example-begin radix-sort-pairs-descending
            :end-before: example-end radix-sort-pairs-descending
    """


def radix_rank(
    keys,
    items_per_thread: int,
    *,
    begin_bit: int | None = None,
    end_bit: int | None = None,
    descending: bool = False,
    temp_storage=None,
):
    """Block-wide radix rank for keys.

    Example:
        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_radix_rank_api.py
            :language: python
            :dedent:
            :start-after: example-begin radix-rank
            :end-before: example-end radix-rank
    """


def run_length(
    items,
    items_per_thread: int,
    *,
    temp_storage=None,
    total_decoded_size=None,
    relative_offsets=None,
    window_offset: int | None = None,
):
    """Block-wide run-length decode for per-thread items.

    Example:
        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_run_length_api.py
            :language: python
            :dedent:
            :start-after: example-begin run-length
            :end-before: example-end run-length
    """


def histogram(
    samples,
    histogram_output,
    *,
    bins: int | None = None,
    temp_storage=None,
):
    """Block-wide histogram (single-phase uses ``samples`` + output).

    Example:
        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_histogram_api.py
            :language: python
            :dedent:
            :start-after: example-begin histogram
            :end-before: example-end histogram
    """


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
