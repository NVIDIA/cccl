# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CUB's block and warp primitives for kernels written in the CuTe DSL.

Import the package and call its functions inside a kernel:

.. code-block:: python

   import cuda.coop.cutlass as coop

`Full example on GitHub
<https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/examples/prefix_sum.py>`__

.. literalinclude::
    ../../../python/cuda_coop_cutlass/examples/prefix_sum.py
    :language: python
    :start-after: docs: start cutlass-prefix-sum
    :end-before: docs: end cutlass-prefix-sum

Every primitive is called the same way:

* The first argument is a :ref:`thread group <coop-thread-groups>`, the
  threads that make the call together.
* The operands follow by position: the memory to read or write, and each
  thread's items as a :ref:`ThreadData <coop-thread-data>`.
* Controls follow by keyword.

Each call is single-phase: there is nothing to construct first. The block
size comes from the launch, the element type from the memory or the items,
and the number of items from ``items_per_thread`` or the payload. Scratch
space in shared memory comes from the compiler, or from a
:ref:`TempStorage <coop-temp-storage>` that the kernel passes.
:ref:`Single-phase calls <coop-concepts-single-phase>` shows the same kernel
written with CUB's C++ classes.

Controls that select code are Python values, fixed when the kernel is
compiled: ``items_per_thread``, ``algorithm``, operators, the group, and
the options of a ``TempStorage``. A kernel receives such a value as an
argument declared ``cutlass.Constexpr``, as ``items_per_thread`` is above.
Controls that describe data can also be CuTe values, read while the kernel
runs: ``valid_items``, ``offset``, ``oob_default`` and a scan's
``initial_value`` are examples. Each entry says what its controls accept.

A runtime loop can repeat one call. A ``cutlass.range_constexpr`` loop is
unrolled, so each trip can make a different call.

Primitives work in a kernel and in any ``@cute.jit`` function that the
kernel calls. When CuTe finishes tracing, the package compiles the CUB code
for every call it traced in one NVRTC compilation, and CuTe links the
result into the kernel. The compiled code is kept in a cache on disk,
described in :doc:`configuration`. To launch a kernel more than once,
compile it with ``cute.compile`` and keep the callable it returns.

The package includes ``.pyi`` stubs for type checkers.
"""

from ._compiler._runtime import validate_cutlass_runtime
from ._device_operator import DeviceOperator, compile_operator
from ._group_exchange import exchange
from ._group_histogram import histogram
from ._group_load_store import load, store
from ._group_merge_sort import merge_sort_keys, merge_sort_pairs
from ._group_neighbors import adjacent_difference, discontinuity
from ._group_radix_sort import (
    radix_rank_keys,
    radix_sort_keys,
    radix_sort_pairs,
)
from ._group_reduce import reduce, sum
from ._group_reduce_batched import reduce_batched
from ._group_run_length import run_length_decode, run_length_decode_into
from ._group_scan import (
    exclusive_scan,
    exclusive_sum,
    inclusive_scan,
    inclusive_sum,
    scan,
)
from ._group_shuffle import shuffle
from ._group_topk import (
    topk_max_keys,
    topk_max_pairs,
    topk_min_keys,
    topk_min_pairs,
)
from ._temp_storage import TempStorage
from ._thread_data import (
    ThreadData,
    ThreadDataLoadSource,
    ThreadDataSource,
    ThreadDataTensorMetadata,
)
from ._thread_group import (
    Hierarchy,
    ThreadGroup,
    ThreadHierarchy,
    this_block,
    this_cluster,
    this_grid,
    this_thread,
    this_warp,
)
from ._typing import TempStorageLike, ThreadDataLike

__all__ = [
    "DeviceOperator",
    "compile_operator",
    "reduce_batched",
    "run_length_decode",
    "run_length_decode_into",
    "histogram",
    "adjacent_difference",
    "discontinuity",
    "topk_min_keys",
    "topk_max_keys",
    "topk_min_pairs",
    "topk_max_pairs",
    "radix_sort_keys",
    "radix_sort_pairs",
    "radix_rank_keys",
    "merge_sort_keys",
    "merge_sort_pairs",
    "exchange",
    "shuffle",
    "scan",
    "exclusive_scan",
    "inclusive_scan",
    "exclusive_sum",
    "inclusive_sum",
    "reduce",
    "sum",
    "load",
    "store",
    "Hierarchy",
    "ThreadGroup",
    "ThreadHierarchy",
    "ThreadData",
    "ThreadDataLoadSource",
    "ThreadDataSource",
    "ThreadDataTensorMetadata",
    "TempStorage",
    "TempStorageLike",
    "ThreadDataLike",
    "this_block",
    "this_cluster",
    "this_grid",
    "this_thread",
    "this_warp",
]


def __dir__():
    """List the public API for interactive completion."""
    return sorted(__all__)


validate_cutlass_runtime()
del validate_cutlass_runtime
