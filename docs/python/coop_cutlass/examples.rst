.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Executable examples
===================

Run these programs from a compatible installed environment. Each example
allocates device inputs, compiles and launches a CuTe kernel, and checks its
defined results against a CPU reference. The :doc:`Programming Guide
<programming_guide>` explains their layout and synchronization requirements.

.. list-table:: Examples
   :header-rows: 1
   :widths: 28 72

   * - Program
     - Demonstrates
   * - :download:`scan.py <../../../python/cuda_coop_cutlass/examples/scan.py>`
     - Block prefix sums with scratch reuse and a nonzero initial value.
   * - :download:`reduce.py <../../../python/cuda_coop_cutlass/examples/reduce.py>`
     - Block and logical-warp reductions with defined leader results.
   * - :download:`exchange_shuffle.py <../../../python/cuda_coop_cutlass/examples/exchange_shuffle.py>`
     - Blocked/striped movement, scatter, and scalar/array Shuffle.
   * - :download:`merge_sort.py <../../../python/cuda_coop_cutlass/examples/merge_sort.py>`
     - Partial-tile key and pair sorting with input preservation.
   * - :download:`radix.py <../../../python/cuda_coop_cutlass/examples/radix.py>`
     - Radix Sort, Rank, striped output, and bin prefixes.
   * - :download:`topk.py <../../../python/cuda_coop_cutlass/examples/topk.py>`
     - Minimum keys and maximum pairs without assumptions about ties or order.
   * - :download:`reduce_batched.py <../../../python/cuda_coop_cutlass/examples/reduce_batched.py>`
     - Independent per-feature warp reductions and output placement.
   * - :download:`shared_staging.py <../../../python/cuda_coop_cutlass/examples/shared_staging.py>`
     - Native CuTe staging, explicit synchronization, and returning Load from shared memory.

The :download:`debugger walkthrough <debugger_walkthrough.py>` compares
Direct and Transpose tile copies. It compiles once and executes the compiled
callable twice; see :ref:`cuda.coop.cutlass.debugger_walkthrough`.
