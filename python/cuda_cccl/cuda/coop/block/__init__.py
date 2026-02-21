# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os

if os.environ.get("CCCL_COOP_DOCS") == "1":
    from .api import *  # noqa: F403
    from .api import __all__  # noqa: F401
else:
    from ._block_adjacent_difference import (
        BlockAdjacentDifferenceType,
        adjacent_difference,
    )
    from ._block_discontinuity import (
        BlockDiscontinuityType,
        discontinuity,
    )
    from ._block_exchange import (
        BlockExchangeType,
        exchange,
    )
    from ._block_histogram import histogram
    from ._block_load_store import (
        load,
        store,
    )
    from ._block_merge_sort import merge_sort_keys, merge_sort_pairs
    from ._block_radix_rank import radix_rank
    from ._block_radix_sort import (
        radix_sort_keys,
        radix_sort_keys_descending,
        radix_sort_pairs,
        radix_sort_pairs_descending,
    )
    from ._block_reduce import reduce, sum
    from ._block_run_length_decode import run_length
    from ._block_scan import (
        exclusive_scan,
        exclusive_sum,
        inclusive_scan,
        inclusive_sum,
        scan,
    )
    from ._block_shuffle import (
        BlockShuffleType,
        shuffle,
    )

    def _normalize_threads_per_block(kwargs, threads_per_block):
        kw = dict(kwargs)
        if threads_per_block is None:
            threads_per_block = kw.pop("dim", None)
        if threads_per_block is not None:
            kw["threads_per_block"] = threads_per_block
        return kw

    def _normalize_scan_prefix(kwargs, prefix_op, target_name):
        kw = dict(kwargs)
        if prefix_op is None:
            prefix_op = kw.pop("prefix_op", None)
        if prefix_op is None:
            prefix_op = kw.pop("block_prefix_callback_op", None)
        if prefix_op is not None:
            kw[target_name] = prefix_op
        return kw

    # Maker-style factory functions.
    #
    # These are the public two-phase entry points. They return Invocable
    # objects (or parent primitive objects for stateful APIs like histogram
    # and run_length). The wrappers intentionally accept both
    # `threads_per_block=` and `dim=` keyword forms.
    def make_load(
        dtype,
        threads_per_block=None,
        items_per_thread=1,
        algorithm="direct",
        **kwargs,
    ):
        kw = _normalize_threads_per_block(kwargs, threads_per_block)
        return load.create(
            dtype=dtype,
            items_per_thread=items_per_thread,
            algorithm=algorithm,
            **kw,
        )

    def make_store(
        dtype,
        threads_per_block=None,
        items_per_thread=1,
        algorithm="direct",
        **kwargs,
    ):
        kw = _normalize_threads_per_block(kwargs, threads_per_block)
        return store.create(
            dtype=dtype,
            items_per_thread=items_per_thread,
            algorithm=algorithm,
            **kw,
        )

    def make_exchange(
        dtype,
        threads_per_block=None,
        items_per_thread=1,
        block_exchange_type=BlockExchangeType.StripedToBlocked,
        **kwargs,
    ):
        kw = _normalize_threads_per_block(kwargs, threads_per_block)
        return exchange.create(
            block_exchange_type=block_exchange_type,
            dtype=dtype,
            items_per_thread=items_per_thread,
            **kw,
        )

    def make_merge_sort_keys(
        dtype,
        threads_per_block=None,
        items_per_thread=1,
        compare_op=None,
        **kwargs,
    ):
        kw = _normalize_threads_per_block(kwargs, threads_per_block)
        return merge_sort_keys.create(
            dtype=dtype,
            items_per_thread=items_per_thread,
            compare_op=compare_op,
            **kw,
        )

    def make_merge_sort_pairs(
        keys,
        values,
        threads_per_block=None,
        items_per_thread=1,
        compare_op=None,
        **kwargs,
    ):
        kw = _normalize_threads_per_block(kwargs, threads_per_block)
        return merge_sort_pairs.create(
            keys=keys,
            values=values,
            items_per_thread=items_per_thread,
            compare_op=compare_op,
            **kw,
        )

    def make_radix_sort_keys(
        dtype,
        threads_per_block=None,
        items_per_thread=1,
        **kwargs,
    ):
        kw = _normalize_threads_per_block(kwargs, threads_per_block)
        return radix_sort_keys.create(
            dtype=dtype,
            items_per_thread=items_per_thread,
            **kw,
        )

    def make_radix_sort_keys_descending(
        dtype,
        threads_per_block=None,
        items_per_thread=1,
        **kwargs,
    ):
        kw = _normalize_threads_per_block(kwargs, threads_per_block)
        return radix_sort_keys_descending.create(
            dtype=dtype,
            items_per_thread=items_per_thread,
            **kw,
        )

    def make_radix_sort_pairs(
        keys,
        values,
        threads_per_block=None,
        items_per_thread=1,
        **kwargs,
    ):
        kw = _normalize_threads_per_block(kwargs, threads_per_block)
        return radix_sort_pairs.create(
            keys=keys,
            values=values,
            items_per_thread=items_per_thread,
            **kw,
        )

    def make_radix_sort_pairs_descending(
        keys,
        values,
        threads_per_block=None,
        items_per_thread=1,
        **kwargs,
    ):
        kw = _normalize_threads_per_block(kwargs, threads_per_block)
        return radix_sort_pairs_descending.create(
            keys=keys,
            values=values,
            items_per_thread=items_per_thread,
            **kw,
        )

    def make_radix_rank(
        dtype,
        threads_per_block=None,
        items_per_thread=1,
        begin_bit=0,
        end_bit=None,
        **kwargs,
    ):
        kw = _normalize_threads_per_block(kwargs, threads_per_block)
        return radix_rank.create(
            dtype=dtype,
            items_per_thread=items_per_thread,
            begin_bit=begin_bit,
            end_bit=end_bit,
            **kw,
        )

    def make_reduce(
        dtype,
        threads_per_block=None,
        binary_op=None,
        items_per_thread=1,
        algorithm="warp_reductions",
        **kwargs,
    ):
        kw = _normalize_threads_per_block(kwargs, threads_per_block)
        return reduce.create(
            dtype=dtype,
            binary_op=binary_op,
            items_per_thread=items_per_thread,
            algorithm=algorithm,
            **kw,
        )

    def make_sum(
        dtype,
        threads_per_block=None,
        items_per_thread=1,
        algorithm="warp_reductions",
        **kwargs,
    ):
        kw = _normalize_threads_per_block(kwargs, threads_per_block)
        return sum.create(
            dtype=dtype,
            items_per_thread=items_per_thread,
            algorithm=algorithm,
            **kw,
        )

    def make_scan(
        dtype,
        threads_per_block=None,
        items_per_thread=1,
        initial_value=None,
        mode="exclusive",
        scan_op="+",
        block_prefix_callback_op=None,
        **kwargs,
    ):
        kw = _normalize_threads_per_block(kwargs, threads_per_block)
        kw = _normalize_scan_prefix(
            kw,
            block_prefix_callback_op,
            target_name="block_prefix_callback_op",
        )
        return scan.create(
            dtype=dtype,
            items_per_thread=items_per_thread,
            initial_value=initial_value,
            mode=mode,
            scan_op=scan_op,
            **kw,
        )

    def make_exclusive_sum(
        dtype,
        threads_per_block=None,
        items_per_thread=1,
        prefix_op=None,
        **kwargs,
    ):
        kw = _normalize_threads_per_block(kwargs, threads_per_block)
        kw = _normalize_scan_prefix(kw, prefix_op, target_name="prefix_op")
        return exclusive_sum.create(
            dtype=dtype,
            items_per_thread=items_per_thread,
            **kw,
        )

    def make_inclusive_sum(
        dtype,
        threads_per_block=None,
        items_per_thread=1,
        prefix_op=None,
        **kwargs,
    ):
        kw = _normalize_threads_per_block(kwargs, threads_per_block)
        kw = _normalize_scan_prefix(kw, prefix_op, target_name="prefix_op")
        return inclusive_sum.create(
            dtype=dtype,
            items_per_thread=items_per_thread,
            **kw,
        )

    def make_exclusive_scan(
        dtype,
        threads_per_block=None,
        scan_op="+",
        items_per_thread=1,
        initial_value=None,
        prefix_op=None,
        **kwargs,
    ):
        kw = _normalize_threads_per_block(kwargs, threads_per_block)
        kw = _normalize_scan_prefix(kw, prefix_op, target_name="prefix_op")
        return exclusive_scan.create(
            dtype=dtype,
            scan_op=scan_op,
            items_per_thread=items_per_thread,
            initial_value=initial_value,
            **kw,
        )

    def make_inclusive_scan(
        dtype,
        threads_per_block=None,
        scan_op="+",
        items_per_thread=1,
        initial_value=None,
        prefix_op=None,
        **kwargs,
    ):
        kw = _normalize_threads_per_block(kwargs, threads_per_block)
        kw = _normalize_scan_prefix(kw, prefix_op, target_name="prefix_op")
        return inclusive_scan.create(
            dtype=dtype,
            scan_op=scan_op,
            items_per_thread=items_per_thread,
            initial_value=initial_value,
            **kw,
        )

    def make_histogram(
        item_dtype,
        counter_dtype,
        threads_per_block=None,
        items_per_thread=1,
        **kwargs,
    ):
        kw = dict(kwargs)
        if threads_per_block is None:
            threads_per_block = kw.pop("dim", None)
        if threads_per_block is not None:
            kw["dim"] = threads_per_block
        return histogram.create(
            item_dtype=item_dtype,
            counter_dtype=counter_dtype,
            items_per_thread=items_per_thread,
            **kw,
        )

    def make_run_length(
        item_dtype,
        threads_per_block=None,
        runs_per_thread=1,
        decoded_items_per_thread=1,
        **kwargs,
    ):
        kw = dict(kwargs)
        if threads_per_block is None:
            threads_per_block = kw.pop("dim", None)
        if threads_per_block is not None:
            kw["dim"] = threads_per_block
        return run_length.create(
            item_dtype=item_dtype,
            runs_per_thread=runs_per_thread,
            decoded_items_per_thread=decoded_items_per_thread,
            **kw,
        )

    def make_adjacent_difference(
        dtype,
        threads_per_block=None,
        items_per_thread=1,
        difference_op=None,
        **kwargs,
    ):
        kw = _normalize_threads_per_block(kwargs, threads_per_block)
        return adjacent_difference.create(
            dtype=dtype,
            items_per_thread=items_per_thread,
            difference_op=difference_op,
            **kw,
        )

    def make_discontinuity(
        dtype,
        threads_per_block=None,
        items_per_thread=1,
        flag_op=None,
        **kwargs,
    ):
        kw = _normalize_threads_per_block(kwargs, threads_per_block)
        return discontinuity.create(
            dtype=dtype,
            items_per_thread=items_per_thread,
            flag_op=flag_op,
            **kw,
        )

    def make_shuffle(
        dtype,
        threads_per_block=None,
        items_per_thread=1,
        **kwargs,
    ):
        kw = _normalize_threads_per_block(kwargs, threads_per_block)
        return shuffle.create(
            dtype=dtype,
            items_per_thread=items_per_thread,
            **kw,
        )

    __all__ = [
        "BlockExchangeType",
        "BlockDiscontinuityType",
        "BlockAdjacentDifferenceType",
        "BlockShuffleType",
        "adjacent_difference",
        "shuffle",
        "exchange",
        "discontinuity",
        "exclusive_scan",
        "exclusive_sum",
        "histogram",
        "inclusive_scan",
        "inclusive_sum",
        "load",
        "merge_sort_keys",
        "merge_sort_pairs",
        "radix_sort_keys",
        "radix_sort_keys_descending",
        "radix_sort_pairs",
        "radix_sort_pairs_descending",
        "radix_rank",
        "reduce",
        "run_length",
        "scan",
        "store",
        "sum",
        "make_adjacent_difference",
        "make_discontinuity",
        "make_exchange",
        "make_exclusive_scan",
        "make_exclusive_sum",
        "make_histogram",
        "make_inclusive_scan",
        "make_inclusive_sum",
        "make_load",
        "make_merge_sort_keys",
        "make_merge_sort_pairs",
        "make_radix_rank",
        "make_radix_sort_keys",
        "make_radix_sort_keys_descending",
        "make_radix_sort_pairs",
        "make_radix_sort_pairs_descending",
        "make_reduce",
        "make_run_length",
        "make_scan",
        "make_shuffle",
        "make_store",
        "make_sum",
    ]
