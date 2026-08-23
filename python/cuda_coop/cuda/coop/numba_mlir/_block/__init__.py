# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Planner-private block providers for Numba-CUDA-MLIR."""

from ._block_adjacent_difference import (
    BlockAdjacentDifferenceType as BlockAdjacentDifferenceType,
)
from ._block_adjacent_difference import adjacent_difference as adjacent_difference
from ._block_discontinuity import (
    BlockDiscontinuityType as BlockDiscontinuityType,
)
from ._block_discontinuity import discontinuity as discontinuity
from ._block_exchange import BlockExchangeType as BlockExchangeType
from ._block_exchange import exchange as exchange
from ._block_load_store import load as load
from ._block_load_store import store as store
from ._block_merge_sort import merge_sort_keys as merge_sort_keys
from ._block_merge_sort import merge_sort_pairs as merge_sort_pairs
from ._block_radix_rank import radix_rank as radix_rank
from ._block_radix_sort import radix_sort_keys as radix_sort_keys
from ._block_radix_sort import (
    radix_sort_keys_descending as radix_sort_keys_descending,
)
from ._block_radix_sort import radix_sort_pairs as radix_sort_pairs
from ._block_radix_sort import (
    radix_sort_pairs_descending as radix_sort_pairs_descending,
)
from ._block_reduce import block_reduce_builtin as block_reduce_builtin
from ._block_reduce import reduce as reduce
from ._block_reduce import sum as sum
from ._block_scan import scan as scan
from ._block_shuffle import BlockShuffleType as BlockShuffleType
from ._block_shuffle import shuffle as shuffle

__all__: tuple[str, ...] = ()
