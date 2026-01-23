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


def exclusive_sum(
    items,
    items_per_thread: int,
    *,
    prefix_op: Callable | None = None,
    algorithm: str | None = None,
    temp_storage=None,
):
    """Exclusive warp-wide sum for per-thread items.

    Example:
        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_warp_scan_api.py
            :language: python
            :dedent:
            :start-after: example-begin exclusive-sum
            :end-before: example-end exclusive-sum
    """


def inclusive_sum(
    items,
    items_per_thread: int,
    *,
    prefix_op: Callable | None = None,
    algorithm: str | None = None,
    temp_storage=None,
):
    """Inclusive warp-wide sum for per-thread items.

    Example:
        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_warp_scan_api.py
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
):
    """Exclusive warp-wide scan with a custom operator.

    Example:
        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_warp_scan_api.py
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
):
    """Inclusive warp-wide scan with a custom operator.

    Example:
        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_warp_scan_api.py
            :language: python
            :dedent:
            :start-after: example-begin inclusive-scan
            :end-before: example-end inclusive-scan
    """


def reduce(
    items,
    items_per_thread: int = 1,
    *,
    binary_op: Callable | None = None,
    algorithm: str | None = None,
    temp_storage=None,
):
    """Warp-wide reduction with a custom operator.

    Example:
        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_warp_reduce_api.py
            :language: python
            :dedent:
            :start-after: example-begin reduce
            :end-before: example-end reduce

        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_warp_reduce_api.py
            :language: python
            :dedent:
            :start-after: example-begin reduce-valid-items
            :end-before: example-end reduce-valid-items
    """


def sum(
    items,
    items_per_thread: int = 1,
    *,
    algorithm: str | None = None,
    temp_storage=None,
):
    """Warp-wide sum reduction.

    Example:
        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_warp_reduce_api.py
            :language: python
            :dedent:
            :start-after: example-begin sum
            :end-before: example-end sum

        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_warp_reduce_api.py
            :language: python
            :dedent:
            :start-after: example-begin sum-valid-items
            :end-before: example-end sum-valid-items
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
    """Warp-wide cooperative load into per-thread items.

    Example:
        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_warp_load_store_api.py
            :language: python
            :dedent:
            :start-after: example-begin load-store
            :end-before: example-end load-store

        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_warp_load_store_api.py
            :language: python
            :dedent:
            :start-after: example-begin load-store-num-valid-oob-default
            :end-before: example-end load-store-num-valid-oob-default
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
    """Warp-wide cooperative store from per-thread items.

    Example:
        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_warp_load_store_api.py
            :language: python
            :dedent:
            :start-after: example-begin load-store
            :end-before: example-end load-store
    """


def exchange(
    items,
    *,
    items_per_thread: int,
    warp_exchange_type: WarpExchangeType,
    ranks=None,
    valid_flags=None,
    temp_storage=None,
):
    """Rearrange items across a warp using an exchange pattern.

    Example:
        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_warp_exchange_api.py
            :language: python
            :dedent:
            :start-after: example-begin striped-to-blocked
            :end-before: example-end striped-to-blocked

        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_warp_exchange_api.py
            :language: python
            :dedent:
            :start-after: example-begin blocked-to-striped
            :end-before: example-end blocked-to-striped
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
    """Warp-wide merge sort for keys.

    Example:
        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_warp_merge_sort_api.py
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
    """Warp-wide merge sort for key-value pairs.

    Example:
        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_warp_merge_sort_pairs_api.py
            :language: python
            :dedent:
            :start-after: example-begin merge-sort-pairs
            :end-before: example-end merge-sort-pairs
    """


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
