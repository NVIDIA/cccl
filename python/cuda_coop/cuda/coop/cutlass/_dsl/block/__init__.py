# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Block-scoped compatibility primitives for CuTe DSL kernels.

The functions in this module lower through provider-generated LTO-IR shims and
are single-phase from Python's perspective. Full ``sum`` and ``reduce`` calls
delegate to the same broadcasted CUDAX artifact as the preferred group-first
frontend. Partial counts and explicit block algorithms select direct CUB and
return a value defined only at block rank zero.

Examples:

.. code-block:: python

   import cuda.coop.cutlass as coop

   total = coop.reduce(coop.this_block(), value)
   keys, values = coop._block.radix_sort_pairs(keys, values, begin_bit=0, end_bit=8)

The scoped surface remains for compatibility and advanced selectors. Use
``TempStorage`` when supported collectives should share explicitly sized
scratch. Reduce accepts legacy block ``TempStorage`` syntax but its shared
CUDAX/CUB plan owns the actual scratch and does not charge that object.
"""

from . import _api as _api
from . import _factory as _factory
from ._difference import (
    BlockAdjacentDifferenceType,
    adjacent_difference,
    adjacent_difference_subtract_left,
    adjacent_difference_subtract_right,
)
from ._discontinuity import (
    BlockDiscontinuityType,
    discontinuity,
    discontinuity_flag_heads,
    discontinuity_flag_heads_and_tails,
    discontinuity_flag_tails,
)
from ._exchange import (
    BlockExchangeType,
    exchange,
    exchange_blocked_to_striped,
    exchange_blocked_to_warp_striped,
    exchange_scatter_to_blocked,
    exchange_scatter_to_striped,
    exchange_scatter_to_striped_flagged,
    exchange_scatter_to_striped_guarded,
    exchange_striped_to_blocked,
    exchange_warp_striped_to_blocked,
)
from ._load_store import load, store
from ._reduce import reduce, row_sum, sum
from ._scan import exclusive_scan, exclusive_sum, inclusive_scan, inclusive_sum, scan
from ._shuffle import (
    BlockShuffleType,
    shuffle,
    shuffle_down,
    shuffle_offset,
    shuffle_rotate,
    shuffle_up,
)
from ._single_phase import TempStorage

# Secondary compatibility adapters remain available through the private block
# module. Its wrappers preserve the private import scope reported by deferred
# factories.
make_adjacent_difference = _factory.make_adjacent_difference
make_discontinuity = _factory.make_discontinuity
make_exchange = _factory.make_exchange
make_exclusive_scan = _factory.make_exclusive_scan
make_exclusive_sum = _factory.make_exclusive_sum
make_inclusive_scan = _factory.make_inclusive_scan
make_inclusive_sum = _factory.make_inclusive_sum
make_load = _factory.make_load
make_reduce = _factory.make_reduce
make_scan = _factory.make_scan
make_shuffle = _factory.make_shuffle
make_store = _factory.make_store
make_sum = _factory.make_sum

__all__ = [
    "BlockAdjacentDifferenceType",
    "BlockDiscontinuityType",
    "BlockExchangeType",
    "BlockShuffleType",
    "TempStorage",
    "adjacent_difference",
    "adjacent_difference_subtract_left",
    "adjacent_difference_subtract_right",
    "discontinuity",
    "discontinuity_flag_heads",
    "discontinuity_flag_heads_and_tails",
    "discontinuity_flag_tails",
    "exchange",
    "exchange_blocked_to_striped",
    "exchange_blocked_to_warp_striped",
    "exchange_scatter_to_blocked",
    "exchange_scatter_to_striped",
    "exchange_scatter_to_striped_flagged",
    "exchange_scatter_to_striped_guarded",
    "exchange_striped_to_blocked",
    "exchange_warp_striped_to_blocked",
    "exclusive_scan",
    "exclusive_sum",
    "inclusive_scan",
    "inclusive_sum",
    "load",
    "make_adjacent_difference",
    "make_discontinuity",
    "make_exchange",
    "make_exclusive_scan",
    "make_exclusive_sum",
    "make_inclusive_scan",
    "make_inclusive_sum",
    "make_load",
    "make_reduce",
    "make_scan",
    "make_shuffle",
    "make_store",
    "make_sum",
    "reduce",
    "row_sum",
    "scan",
    "shuffle",
    "shuffle_down",
    "shuffle_offset",
    "shuffle_rotate",
    "shuffle_up",
    "store",
    "sum",
]
