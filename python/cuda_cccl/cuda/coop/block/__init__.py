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
    from ._block_adjacent_difference import (
        _make_adjacent_difference_two_phase as make_adjacent_difference,
    )
    from ._block_discontinuity import (
        BlockDiscontinuityType,
        discontinuity,
    )
    from ._block_discontinuity import (
        _make_discontinuity_two_phase as make_discontinuity,
    )
    from ._block_exchange import (
        BlockExchangeType,
        exchange,
    )
    from ._block_exchange import (
        _make_exchange_two_phase as make_exchange,
    )
    from ._block_histogram import (
        _make_histogram_two_phase as make_histogram,
    )
    from ._block_histogram import (
        histogram,
    )
    from ._block_load_store import (
        _make_load_two_phase as make_load,
    )
    from ._block_load_store import (
        _make_store_two_phase as make_store,
    )
    from ._block_load_store import (
        load,
        store,
    )
    from ._block_merge_sort import (
        _make_merge_sort_keys_two_phase as make_merge_sort_keys,
    )
    from ._block_merge_sort import (
        _make_merge_sort_pairs_two_phase as make_merge_sort_pairs,
    )
    from ._block_merge_sort import (
        merge_sort_keys,
        merge_sort_pairs,
    )
    from ._block_radix_rank import (
        _make_radix_rank_two_phase as make_radix_rank,
    )
    from ._block_radix_rank import (
        radix_rank,
    )
    from ._block_radix_sort import (
        _make_radix_sort_keys_descending_two_phase as make_radix_sort_keys_descending,
    )
    from ._block_radix_sort import (
        _make_radix_sort_keys_two_phase as make_radix_sort_keys,
    )
    from ._block_radix_sort import (
        _make_radix_sort_pairs_descending_two_phase as make_radix_sort_pairs_descending,
    )
    from ._block_radix_sort import (
        _make_radix_sort_pairs_two_phase as make_radix_sort_pairs,
    )
    from ._block_radix_sort import (
        radix_sort_keys,
        radix_sort_keys_descending,
        radix_sort_pairs,
        radix_sort_pairs_descending,
    )
    from ._block_reduce import (
        _make_reduce_two_phase as make_reduce,
    )
    from ._block_reduce import (
        _make_sum_two_phase as make_sum,
    )
    from ._block_reduce import (
        reduce,
        sum,
    )
    from ._block_run_length_decode import (
        _make_run_length_two_phase as make_run_length,
    )
    from ._block_run_length_decode import (
        run_length,
    )
    from ._block_scan import (
        _make_exclusive_scan_two_phase as make_exclusive_scan,
    )
    from ._block_scan import (
        _make_exclusive_sum_two_phase as make_exclusive_sum,
    )
    from ._block_scan import (
        _make_inclusive_scan_two_phase as make_inclusive_scan,
    )
    from ._block_scan import (
        _make_inclusive_sum_two_phase as make_inclusive_sum,
    )
    from ._block_scan import (
        _make_scan_two_phase as make_scan,
    )
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
    from ._block_shuffle import (
        _make_shuffle_two_phase as make_shuffle,
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
