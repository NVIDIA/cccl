.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Executable examples
===================

Read these programs on GitHub, then run them from a compatible installed
environment. Each example allocates device inputs, compiles and launches a
CuTe kernel, and checks its defined results against a CPU reference. The
:doc:`Programming Guide <programming_guide>` and the
:doc:`API reference <api>` explain their layout and synchronization
requirements.

.. contents:: Programs
   :local:
   :depth: 1

scan.py
-------

Block prefix sums with scratch reuse and a nonzero initial value.

`View scan.py on GitHub
<https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/examples/scan.py>`_

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/scan.py
   :language: python

reduce.py
---------

The sum of every item in a block, the largest ``payload[0]`` in each group
of eight lanes, and the sum of ``payload[0]`` over the first 23 threads.
:ref:`Reduction <coop-cutlass-reduce>` in the API reference explains each
call.

`View reduce.py on GitHub
<https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/examples/reduce.py>`_

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/reduce.py
   :language: python

exchange_shuffle.py
-------------------

A blocked-to-striped ``exchange``, then a ``shuffle`` down by one position.

`View exchange_shuffle.py on GitHub
<https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/examples/exchange_shuffle.py>`_

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/exchange_shuffle.py
   :language: python

merge_sort.py
-------------

Partial-tile key and pair sorting with input preservation.

`View merge_sort.py on GitHub
<https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/examples/merge_sort.py>`_

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/merge_sort.py
   :language: python

radix.py
--------

Radix Sort, Rank, striped output, and bin prefixes.

`View radix.py on GitHub
<https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/examples/radix.py>`_

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/radix.py
   :language: python

topk.py
-------

Minimum keys and maximum pairs without assumptions about ties or order.

`View topk.py on GitHub
<https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/examples/topk.py>`_

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/topk.py
   :language: python

reduce_batched.py
-----------------

One sum for each feature across each warp.
:ref:`Batched reduction <coop-cutlass-reduce-batched>` in the API reference
explains which lane holds each sum.

`View reduce_batched.py on GitHub
<https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/examples/reduce_batched.py>`_

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/reduce_batched.py
   :language: python

shared_staging.py
-----------------

Native CuTe staging, explicit synchronization, and Load from shared memory.

`View shared_staging.py on GitHub
<https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/examples/shared_staging.py>`_

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/shared_staging.py
   :language: python

.. _tensor-memory-topk-py:

tensor_memory_topk.py
---------------------

Tensor Core MMA, cooperative TopK in registers, and a TMEM round trip across
dependent windows. Requires an SM100 or SM103 GPU. The
:doc:`Tensor Core and TopK walkthrough <tensor_memory_topk>` explains the
copy layouts, synchronization, and accumulator feedback.

`View tensor_memory_topk.py on GitHub
<https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/examples/tensor_memory_topk.py>`_

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/tensor_memory_topk.py
   :language: python

Integration walkthroughs
------------------------

The :ref:`debugger walkthrough <cuda.coop.cutlass.debugger_walkthrough>`
compares Direct and Transpose tile copies. It compiles once and executes
the compiled callable twice, with code excerpts and a
`complete debugger_walkthrough.py script on GitHub
<https://github.com/NVIDIA/cccl/blob/main/docs/python/coop_cutlass/debugger_walkthrough.py>`_.

The :doc:`minimal CuTe DSL integration example <cutlass_integration_example>`
calls ``cub::BlockReduce`` from a CuTe kernel using the DSL's own interfaces.
Its walkthrough explains each step with code excerpts and offers the
`complete cutlass_integration_example.py script on GitHub
<https://github.com/NVIDIA/cccl/blob/main/docs/python/coop_cutlass/cutlass_integration_example.py>`_.
