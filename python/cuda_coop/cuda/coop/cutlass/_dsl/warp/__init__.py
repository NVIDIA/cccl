# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Warp-scoped compatibility primitives for CuTe DSL kernels.

The functions in this module lower through provider-generated LTO-IR shims and
official CUDAX/CUB calls from a CuTe DSL kernel. Full physical-warp reductions
use broadcasted CUDAX and define the result on every lane. Partial physical
warps and advanced logical subwarps use CUB and define the aggregate only on
lane zero. Warp scans return the per-lane prefix value.

Examples:

.. code-block:: python

   import cuda.coop.cutlass as coop

   warp_total = coop.reduce(coop.this_warp(), value)
   items = coop._warp.load(values, items_per_thread=2, algorithm="striped")
   lane_prefix = coop._warp.exclusive_sum(items[0])
"""

from . import _api as _api
from . import _factory as _factory
from ._load_store import load, store
from ._reduce import max, min, reduce, sum
from ._single_phase import TempStorage

# Secondary compatibility adapters remain available through the private warp
# module. Its wrappers preserve the private import scope reported by deferred
# factories.
make_load = _factory.make_load
make_max = _factory.make_max
make_min = _factory.make_min
make_reduce = _factory.make_reduce
make_store = _factory.make_store
make_sum = _factory.make_sum

__all__ = [
    "TempStorage",
    "load",
    "make_load",
    "make_max",
    "make_min",
    "make_reduce",
    "make_store",
    "make_sum",
    "max",
    "min",
    "reduce",
    "store",
    "sum",
]
