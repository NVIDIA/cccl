# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os

if os.environ.get("CCCL_COOP_DOCS") == "1":
    from .api import *  # noqa: F403
    from .api import __all__  # noqa: F401
else:
    from ._warp_exchange import WarpExchangeType, exchange
    from ._warp_load_store import load, store
    from ._warp_merge_sort import merge_sort_keys, merge_sort_pairs
    from ._warp_reduce import reduce, sum
    from ._warp_scan import (
        exclusive_scan,
        exclusive_sum,
        inclusive_scan,
        inclusive_sum,
    )

    # Maker-style factory functions.
    #
    # These wrappers are the supported two-phase entry points going forward.
    # They intentionally accept rewrite-internal kwargs as passthrough.
    def make_load(
        dtype,
        items_per_thread=1,
        threads_in_warp=32,
        algorithm=None,
        **kwargs,
    ):
        return load(
            dtype=dtype,
            items_per_thread=items_per_thread,
            threads_in_warp=threads_in_warp,
            algorithm=algorithm,
            **kwargs,
        )

    def make_store(
        dtype,
        items_per_thread=1,
        threads_in_warp=32,
        algorithm=None,
        **kwargs,
    ):
        return store(
            dtype=dtype,
            items_per_thread=items_per_thread,
            threads_in_warp=threads_in_warp,
            algorithm=algorithm,
            **kwargs,
        )

    def make_exchange(
        dtype,
        items_per_thread=1,
        threads_in_warp=32,
        warp_exchange_type=WarpExchangeType.StripedToBlocked,
        **kwargs,
    ):
        return exchange(
            warp_exchange_type=warp_exchange_type,
            dtype=dtype,
            items_per_thread=items_per_thread,
            threads_in_warp=threads_in_warp,
            **kwargs,
        )

    def make_reduce(
        dtype,
        binary_op,
        threads_in_warp=32,
        valid_items=None,
        **kwargs,
    ):
        return reduce(
            dtype=dtype,
            binary_op=binary_op,
            threads_in_warp=threads_in_warp,
            valid_items=valid_items,
            **kwargs,
        )

    def make_sum(
        dtype,
        threads_in_warp=32,
        valid_items=None,
        **kwargs,
    ):
        return sum(
            dtype=dtype,
            threads_in_warp=threads_in_warp,
            valid_items=valid_items,
            **kwargs,
        )

    def make_exclusive_sum(
        dtype,
        threads_in_warp=32,
        **kwargs,
    ):
        return exclusive_sum(
            dtype=dtype,
            threads_in_warp=threads_in_warp,
            **kwargs,
        )

    def make_inclusive_sum(
        dtype,
        threads_in_warp=32,
        **kwargs,
    ):
        return inclusive_sum(
            dtype=dtype,
            threads_in_warp=threads_in_warp,
            **kwargs,
        )

    def make_exclusive_scan(
        dtype,
        scan_op,
        initial_value=None,
        threads_in_warp=32,
        **kwargs,
    ):
        return exclusive_scan(
            dtype=dtype,
            scan_op=scan_op,
            initial_value=initial_value,
            threads_in_warp=threads_in_warp,
            **kwargs,
        )

    def make_inclusive_scan(
        dtype,
        scan_op,
        initial_value=None,
        threads_in_warp=32,
        **kwargs,
    ):
        return inclusive_scan(
            dtype=dtype,
            scan_op=scan_op,
            initial_value=initial_value,
            threads_in_warp=threads_in_warp,
            **kwargs,
        )

    def make_merge_sort_keys(
        dtype,
        items_per_thread,
        compare_op,
        value_dtype=None,
        threads_in_warp=32,
        **kwargs,
    ):
        return merge_sort_keys(
            dtype=dtype,
            items_per_thread=items_per_thread,
            compare_op=compare_op,
            value_dtype=value_dtype,
            threads_in_warp=threads_in_warp,
            **kwargs,
        )

    def make_merge_sort_pairs(
        keys,
        values,
        items_per_thread,
        compare_op,
        threads_in_warp=32,
        **kwargs,
    ):
        return merge_sort_pairs(
            keys=keys,
            values=values,
            items_per_thread=items_per_thread,
            compare_op=compare_op,
            threads_in_warp=threads_in_warp,
            **kwargs,
        )

    __all__ = [
        "exclusive_scan",
        "exclusive_sum",
        "inclusive_scan",
        "inclusive_sum",
        "reduce",
        "sum",
        "merge_sort_keys",
        "merge_sort_pairs",
        "load",
        "store",
        "exchange",
        "WarpExchangeType",
        "make_exchange",
        "make_exclusive_scan",
        "make_exclusive_sum",
        "make_inclusive_scan",
        "make_inclusive_sum",
        "make_load",
        "make_merge_sort_keys",
        "make_merge_sort_pairs",
        "make_reduce",
        "make_store",
        "make_sum",
    ]
