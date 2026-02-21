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
        _make_adjacent_difference_two_phase,
        adjacent_difference,
    )
    from ._block_discontinuity import (
        BlockDiscontinuityType,
        _make_discontinuity_two_phase,
        discontinuity,
    )
    from ._block_exchange import (
        BlockExchangeType,
        _make_exchange_two_phase,
        exchange,
    )
    from ._block_histogram import _make_histogram_two_phase, histogram
    from ._block_load_store import (
        _make_load_two_phase,
        _make_store_two_phase,
        load,
        store,
    )
    from ._block_merge_sort import (
        _make_merge_sort_keys_two_phase,
        _make_merge_sort_pairs_two_phase,
        merge_sort_keys,
        merge_sort_pairs,
    )
    from ._block_radix_rank import _make_radix_rank_two_phase, radix_rank
    from ._block_radix_sort import (
        _make_radix_sort_keys_descending_two_phase,
        _make_radix_sort_keys_two_phase,
        _make_radix_sort_pairs_descending_two_phase,
        _make_radix_sort_pairs_two_phase,
        radix_sort_keys,
        radix_sort_keys_descending,
        radix_sort_pairs,
        radix_sort_pairs_descending,
    )
    from ._block_reduce import (
        _make_reduce_two_phase,
        _make_sum_two_phase,
        reduce,
        sum,
    )
    from ._block_run_length_decode import _make_run_length_two_phase, run_length
    from ._block_scan import (
        _make_exclusive_scan_two_phase,
        _make_exclusive_sum_two_phase,
        _make_inclusive_scan_two_phase,
        _make_inclusive_sum_two_phase,
        _make_scan_two_phase,
        exclusive_scan,
        exclusive_sum,
        inclusive_scan,
        inclusive_sum,
        scan,
    )
    from ._block_shuffle import (
        BlockShuffleType,
        _make_shuffle_two_phase,
        shuffle,
    )

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
        return _make_load_two_phase(
            dtype=dtype,
            threads_per_block=threads_per_block,
            items_per_thread=items_per_thread,
            algorithm=algorithm,
            **kwargs,
        )

    def make_store(
        dtype,
        threads_per_block=None,
        items_per_thread=1,
        algorithm="direct",
        **kwargs,
    ):
        return _make_store_two_phase(
            dtype=dtype,
            threads_per_block=threads_per_block,
            items_per_thread=items_per_thread,
            algorithm=algorithm,
            **kwargs,
        )

    def make_exchange(
        dtype,
        threads_per_block=None,
        items_per_thread=1,
        block_exchange_type=BlockExchangeType.StripedToBlocked,
        **kwargs,
    ):
        return _make_exchange_two_phase(
            block_exchange_type=block_exchange_type,
            dtype=dtype,
            threads_per_block=threads_per_block,
            items_per_thread=items_per_thread,
            **kwargs,
        )

    def make_merge_sort_keys(
        dtype,
        threads_per_block=None,
        items_per_thread=1,
        compare_op=None,
        **kwargs,
    ):
        return _make_merge_sort_keys_two_phase(
            dtype=dtype,
            threads_per_block=threads_per_block,
            items_per_thread=items_per_thread,
            compare_op=compare_op,
            **kwargs,
        )

    def make_merge_sort_pairs(
        keys,
        values,
        threads_per_block=None,
        items_per_thread=1,
        compare_op=None,
        **kwargs,
    ):
        return _make_merge_sort_pairs_two_phase(
            keys=keys,
            values=values,
            threads_per_block=threads_per_block,
            items_per_thread=items_per_thread,
            compare_op=compare_op,
            **kwargs,
        )

    def make_radix_sort_keys(
        dtype,
        threads_per_block=None,
        items_per_thread=1,
        **kwargs,
    ):
        return _make_radix_sort_keys_two_phase(
            dtype=dtype,
            threads_per_block=threads_per_block,
            items_per_thread=items_per_thread,
            **kwargs,
        )

    def make_radix_sort_keys_descending(
        dtype,
        threads_per_block=None,
        items_per_thread=1,
        **kwargs,
    ):
        return _make_radix_sort_keys_descending_two_phase(
            dtype=dtype,
            threads_per_block=threads_per_block,
            items_per_thread=items_per_thread,
            **kwargs,
        )

    def make_radix_sort_pairs(
        keys,
        values,
        threads_per_block=None,
        items_per_thread=1,
        **kwargs,
    ):
        return _make_radix_sort_pairs_two_phase(
            keys=keys,
            values=values,
            threads_per_block=threads_per_block,
            items_per_thread=items_per_thread,
            **kwargs,
        )

    def make_radix_sort_pairs_descending(
        keys,
        values,
        threads_per_block=None,
        items_per_thread=1,
        **kwargs,
    ):
        return _make_radix_sort_pairs_descending_two_phase(
            keys=keys,
            values=values,
            threads_per_block=threads_per_block,
            items_per_thread=items_per_thread,
            **kwargs,
        )

    def make_radix_rank(
        dtype,
        threads_per_block=None,
        items_per_thread=1,
        begin_bit=0,
        end_bit=None,
        **kwargs,
    ):
        return _make_radix_rank_two_phase(
            dtype=dtype,
            threads_per_block=threads_per_block,
            items_per_thread=items_per_thread,
            begin_bit=begin_bit,
            end_bit=end_bit,
            **kwargs,
        )

    def make_reduce(
        dtype,
        threads_per_block=None,
        binary_op=None,
        items_per_thread=1,
        algorithm="warp_reductions",
        **kwargs,
    ):
        return _make_reduce_two_phase(
            dtype=dtype,
            threads_per_block=threads_per_block,
            binary_op=binary_op,
            items_per_thread=items_per_thread,
            algorithm=algorithm,
            **kwargs,
        )

    def make_sum(
        dtype,
        threads_per_block=None,
        items_per_thread=1,
        algorithm="warp_reductions",
        **kwargs,
    ):
        return _make_sum_two_phase(
            dtype=dtype,
            threads_per_block=threads_per_block,
            items_per_thread=items_per_thread,
            algorithm=algorithm,
            **kwargs,
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
        return _make_scan_two_phase(
            dtype=dtype,
            threads_per_block=threads_per_block,
            items_per_thread=items_per_thread,
            initial_value=initial_value,
            mode=mode,
            scan_op=scan_op,
            block_prefix_callback_op=block_prefix_callback_op,
            **kwargs,
        )

    def make_exclusive_sum(
        dtype,
        threads_per_block=None,
        items_per_thread=1,
        prefix_op=None,
        **kwargs,
    ):
        return _make_exclusive_sum_two_phase(
            dtype=dtype,
            threads_per_block=threads_per_block,
            items_per_thread=items_per_thread,
            prefix_op=prefix_op,
            **kwargs,
        )

    def make_inclusive_sum(
        dtype,
        threads_per_block=None,
        items_per_thread=1,
        prefix_op=None,
        **kwargs,
    ):
        return _make_inclusive_sum_two_phase(
            dtype=dtype,
            threads_per_block=threads_per_block,
            items_per_thread=items_per_thread,
            prefix_op=prefix_op,
            **kwargs,
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
        return _make_exclusive_scan_two_phase(
            dtype=dtype,
            threads_per_block=threads_per_block,
            scan_op=scan_op,
            items_per_thread=items_per_thread,
            initial_value=initial_value,
            prefix_op=prefix_op,
            **kwargs,
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
        return _make_inclusive_scan_two_phase(
            dtype=dtype,
            threads_per_block=threads_per_block,
            scan_op=scan_op,
            items_per_thread=items_per_thread,
            initial_value=initial_value,
            prefix_op=prefix_op,
            **kwargs,
        )

    def make_histogram(
        item_dtype,
        counter_dtype,
        threads_per_block=None,
        items_per_thread=1,
        **kwargs,
    ):
        return _make_histogram_two_phase(
            item_dtype=item_dtype,
            counter_dtype=counter_dtype,
            threads_per_block=threads_per_block,
            items_per_thread=items_per_thread,
            **kwargs,
        )

    def make_run_length(
        item_dtype,
        threads_per_block=None,
        runs_per_thread=1,
        decoded_items_per_thread=1,
        **kwargs,
    ):
        return _make_run_length_two_phase(
            item_dtype=item_dtype,
            threads_per_block=threads_per_block,
            runs_per_thread=runs_per_thread,
            decoded_items_per_thread=decoded_items_per_thread,
            **kwargs,
        )

    def make_adjacent_difference(
        dtype,
        threads_per_block=None,
        items_per_thread=1,
        difference_op=None,
        **kwargs,
    ):
        return _make_adjacent_difference_two_phase(
            dtype=dtype,
            threads_per_block=threads_per_block,
            items_per_thread=items_per_thread,
            difference_op=difference_op,
            **kwargs,
        )

    def make_discontinuity(
        dtype,
        threads_per_block=None,
        items_per_thread=1,
        flag_op=None,
        **kwargs,
    ):
        return _make_discontinuity_two_phase(
            dtype=dtype,
            threads_per_block=threads_per_block,
            items_per_thread=items_per_thread,
            flag_op=flag_op,
            **kwargs,
        )

    def make_shuffle(
        dtype,
        threads_per_block=None,
        items_per_thread=1,
        **kwargs,
    ):
        return _make_shuffle_two_phase(
            dtype=dtype,
            threads_per_block=threads_per_block,
            items_per_thread=items_per_thread,
            **kwargs,
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
