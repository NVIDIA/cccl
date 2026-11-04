.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Executable examples
===================

Read or download these programs, then run them from a compatible installed environment. Each example
allocates device inputs, compiles and launches a CuTe kernel, and checks its
defined results against a CPU reference. The :doc:`Programming Guide
<programming_guide>` explains their layout and synchronization requirements.

.. contents:: Programs
   :local:
   :depth: 1

scan.py
-------

Block prefix sums with scratch reuse and a nonzero initial value.

:download:`Download scan.py <../../../python/cuda_coop_cutlass/examples/scan.py>`

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/scan.py
   :language: python

reduce.py
---------

Block and logical-warp reductions with defined leader results.

:download:`Download reduce.py <../../../python/cuda_coop_cutlass/examples/reduce.py>`

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/reduce.py
   :language: python

exchange_shuffle.py
-------------------

Blocked/striped movement, scatter, and scalar/array Shuffle.

:download:`Download exchange_shuffle.py <../../../python/cuda_coop_cutlass/examples/exchange_shuffle.py>`

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/exchange_shuffle.py
   :language: python

merge_sort.py
-------------

Partial-tile key and pair sorting with input preservation.

:download:`Download merge_sort.py <../../../python/cuda_coop_cutlass/examples/merge_sort.py>`

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/merge_sort.py
   :language: python

radix.py
--------

Radix Sort, Rank, striped output, and bin prefixes.

:download:`Download radix.py <../../../python/cuda_coop_cutlass/examples/radix.py>`

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/radix.py
   :language: python

topk.py
-------

Minimum keys and maximum pairs without assumptions about ties or order.

:download:`Download topk.py <../../../python/cuda_coop_cutlass/examples/topk.py>`

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/topk.py
   :language: python

reduce_batched.py
-----------------

Independent per-feature warp reductions and output placement.

:download:`Download reduce_batched.py <../../../python/cuda_coop_cutlass/examples/reduce_batched.py>`

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/reduce_batched.py
   :language: python

shared_staging.py
-----------------

Native CuTe staging, explicit synchronization, and returning Load from shared memory.

:download:`Download shared_staging.py <../../../python/cuda_coop_cutlass/examples/shared_staging.py>`

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/shared_staging.py
   :language: python

Integration walkthroughs
------------------------

The :ref:`debugger walkthrough <cuda.coop.cutlass.debugger_walkthrough>`
compares Direct and Transpose tile copies. It compiles once and executes
the compiled callable twice, with code excerpts and a
:download:`complete script <debugger_walkthrough.py>`.

The :doc:`minimal CuTe DSL integration example <cutlass_integration_example>`
calls ``cub::BlockReduce`` from a CuTe kernel using the DSL's own interfaces.
Its walkthrough explains each step with code excerpts and offers the
:download:`complete script <cutlass_integration_example.py>`.
