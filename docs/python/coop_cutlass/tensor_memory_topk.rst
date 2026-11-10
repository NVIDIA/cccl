.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-tensor-memory-topk:

Tensor Core and TopK interop
============================

This example composes native CuTe Tensor Core operations with
``cuda.coop.cutlass`` selection inside one kernel. Each block owns a
128-by-8 FP32 accumulator in tensor memory (TMEM). In every window, a
Tensor Core matrix multiply-accumulate (MMA) updates it, the threads load it
into registers, and ``coop.topk_max_keys`` selects the largest values.
The threads write the selection back to TMEM, where the next MMA consumes
it as its accumulator input.

.. graphviz::
   :caption: One window; the stored selection becomes the next window's C input.
   :alt: MMA produces a TMEM accumulator. Threads load it into registers,
         apply TopK and zero the tail, then store it to TMEM for the next MMA.

   digraph interop {
       graph [rankdir=TB, bgcolor="#ffffff"];
       node [shape=box, style="rounded,filled", fillcolor="#f4f8ee",
             color="#517a20", fontname="sans-serif", fontsize=12];
       edge [color="#517a20", fontname="sans-serif", fontsize=10];
       mma [label="Tensor Core MMA\nFP16 A and B in shared memory"];
       tmem [label="FP32 accumulator in TMEM"];
       rmem [label="Eight register values per thread"];
       topk [label="Cooperative TopK\nZero the undefined tail"];
       mma -> tmem;
       tmem -> rmem [label="tcgen05 load"];
       rmem -> topk;
       topk -> tmem [label="tcgen05 store"];
       tmem -> mma [label="next window: accumulate onto C"];
   }

Run the example
---------------

Use the :doc:`package prerequisites <index>` and an SM100 or SM103 GPU,
such as B200 or GB300. This program uses the ``tcgen05`` MMA and TMEM
instructions supported by those architectures. Its device check rejects
other architectures, including SM120.

Download :download:`tensor_memory_topk.py
<../../../python/cuda_coop_cutlass/examples/tensor_memory_topk.py>` and run:

.. code-block:: bash

   python tensor_memory_topk.py

The default runs two independent batches, four dependent windows per batch,
and keeps 129 of the 1,024 values after each MMA. The script checks every
window against NumPy and verifies each TMEM store/load round trip exactly.
The :ref:`complete highlighted program <tensor-memory-topk-py>` is also on
the executable examples page.

Storage and participation
-------------------------

All 128 threads participate in the register copies and cooperative TopK.
Warp zero issues the MMA. One elected thread commits its completion to an
``mbarrier``, which all threads wait on before consuming the accumulator.

The kernel's ``SmemAllocator`` allocates the FP16 operands, completion
barrier, and TMEM allocation handle. ``TmemAllocator`` reserves the
accumulator separately. A ``coop.TempStorage`` descriptor supplies TopK's
shared scratch and is reused across windows. CuTe lays out the native shared
allocations and cooperative scratch together.

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/tensor_memory_topk.py
   :language: python
   :start-after: docs: start cutlass-tmem-storage
   :end-before: docs: end cutlass-tmem-storage
   :dedent:

Copy layouts define the register view
-------------------------------------

The TMEM copy partitions the accumulator among threads. Derive the register
shape from the copy's **destination** partition; the TMEM source layout is
not the per-thread register layout. Here each thread receives eight FP32
values. Keeping that layout when copying back preserves the correspondence
between register slots and TMEM positions.

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/tensor_memory_topk.py
   :language: python
   :start-after: docs: start cutlass-tmem-copy-layout
   :end-before: docs: end cutlass-tmem-copy-layout
   :dedent:

MMA produces the next accumulator
---------------------------------

Threads populate A and B in shared memory. The asynchronous-proxy fence
and block barrier precede their consumption by MMA. The first window starts
a fresh accumulator; subsequent windows set ``ACCUMULATE`` and add their
product to the selection left in TMEM. The completion barrier's phase toggles
after each window.

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/tensor_memory_topk.py
   :language: python
   :start-after: docs: start cutlass-tmem-mma
   :end-before: docs: end cutlass-tmem-mma
   :dedent:

SIMT selection operates on registers
------------------------------------

After the TMEM load, ``ThreadData.from_register_tensor`` presents the
register values to ``coop.topk_max_keys``. Selection spans the entire
block's 1,024 values. Its first ``keep`` logical output positions are valid;
their order is unspecified. The code explicitly zeros every other slot
before writing a complete accumulator back to TMEM.

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/tensor_memory_topk.py
   :language: python
   :start-after: docs: start cutlass-tmem-simt
   :end-before: docs: end cutlass-tmem-simt
   :dedent:

TopK packs selected keys into its output prefix. Their original matrix
coordinates are not retained. In this example, the packed selection becomes
the next accumulator state. An algorithm that needs original coordinates
can use ``coop.topk_max_pairs`` with coordinate or index payloads and place
the selected results as its next MMA requires.

Store, verify, and feed the next MMA
------------------------------------

``to_tensor_ssa(like=registers)`` restores the register tensor's layout for
the TMEM store. The store-completion fence and block barrier order that
write before the reload. The reload is saved for validation; its completion
fence and barrier precede the next iteration or TMEM deallocation.

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/tensor_memory_topk.py
   :language: python
   :start-after: docs: start cutlass-tmem-store
   :end-before: docs: end cutlass-tmem-store
   :dedent:

These native synchronization steps cover MMA and TMEM activity. The example
uses ``TempStorage()`` with manual synchronization: the block barriers after
TopK also order scratch reuse before the next window. A scratch-reuse
barrier alone does not complete the native asynchronous operations.

Checking an unordered selection
-------------------------------

The first window uses nonuniform operands so its accumulator contains
different values. Later products add the same value to every accumulator
element. This makes an independent NumPy reference possible without assuming
a particular order for TopK's selected keys: compare sorted multisets before
and after selection, then compare the stored and reloaded values
element-for-element. Each subsequent MMA must incorporate the previous
window's selection for those checks to pass.

The program uses small tiles and records intermediate results to make this
data flow easy to inspect. CUTLASS's `Blackwell GEMM tutorials
<https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm>`_
extend native MMA kernels with TMA transfers, pipelining, and larger tiles.
