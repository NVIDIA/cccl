# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
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
    from ._block_merge_sort import merge_sort_keys
    from ._block_radix_rank import radix_rank
    from ._block_radix_sort import (
        radix_sort_keys,
        radix_sort_keys_descending,
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
        "radix_sort_keys",
        "radix_sort_keys_descending",
        "radix_rank",
        "reduce",
        "run_length",
        "scan",
        "store",
        "sum",
    ]
