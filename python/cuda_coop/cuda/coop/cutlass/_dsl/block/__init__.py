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
from ._load_store import load, store
from ._reduce import reduce, row_sum, sum
from ._scan import exclusive_scan, exclusive_sum, inclusive_scan, inclusive_sum, scan
from ._single_phase import TempStorage

# Secondary compatibility adapters remain available through the private block
# module. Its wrappers preserve the private import scope reported by deferred
# factories.
make_exclusive_scan = _factory.make_exclusive_scan
make_exclusive_sum = _factory.make_exclusive_sum
make_inclusive_scan = _factory.make_inclusive_scan
make_inclusive_sum = _factory.make_inclusive_sum
make_load = _factory.make_load
make_reduce = _factory.make_reduce
make_scan = _factory.make_scan
make_store = _factory.make_store
make_sum = _factory.make_sum

__all__ = [
    "TempStorage",
    "exclusive_scan",
    "exclusive_sum",
    "inclusive_scan",
    "inclusive_sum",
    "load",
    "make_exclusive_scan",
    "make_exclusive_sum",
    "make_inclusive_scan",
    "make_inclusive_sum",
    "make_load",
    "make_reduce",
    "make_scan",
    "make_store",
    "make_sum",
    "reduce",
    "row_sum",
    "scan",
    "store",
    "sum",
]
