.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Counting samples into bins
==========================

``histogram`` counts a complete Block's samples and returns fresh counters.
The bin count is required; defaults select ``i32`` counters, one counter per
thread and the ``Atomic`` algorithm:

.. code-block:: rust

   let counts = coop::histogram(block, &samples,
       coop::HistogramOptions::<32>::default());

Every sample is an integer in ``[0, BINS)``. All input slots participate,
including any fill supplied to Load. Zero-filled tails count toward bin zero.
Unreadable or out-of-range samples trap before native counting. Each call
starts its counters from zero and preserves the input.

A complete program
------------------

This helper loads a tile with an inferred sample type and a generic
``ITEMS_PER_THREAD``. It counts 65 bins using ``Sort`` and returns two
``u64`` counters per thread, together with the preserved input:

.. literalinclude:: ../../rust/cuda-coop/examples/histogram/helper/src/lib.rs
   :language: rust
   :start-after: // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

The host uses 64-thread Blocks and instantiates one and four input items per
thread. Each Block owns 128 output slots: 65 bins and 63 initialized padding
zeros. It checks every counter and the preserved input:

.. literalinclude:: ../../rust/cuda-coop/examples/histogram/src/main.rs
   :language: rust
   :start-after: // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

With the pinned Oxide revision and Rust ``1.99.1``
installed, run from a complete CCCL checkout:

.. code-block:: shell

   cd rust/cuda-coop/examples/histogram
   CUDA_HOME=/usr/local/cuda-13.3 CUDA_PATH=/usr/local/cuda-13.3 \
     cargo oxide run --arch sm_120 --materialize-cubin

Choose the architecture for your GPU. ``auto_shared`` uses the compiler's
plan for the complete temporary-storage requirement.

Independent input and output shapes
-----------------------------------

Samples support ``u8``, ``i32``, ``u32``, ``i64`` and ``u64``. Counters support
``i32``, ``u32``, ``i64`` and ``u64``. ``counter_type::<C>()`` selects their
type and ``bins_per_thread::<K>()`` selects their per-thread extent, independently
of the input's type, item count and alignment. The result is
``ThreadData<C, K>`` with its default alignment, including C's natural alignment.

For a Block of size B, rank t owns bins ``t + i * B`` for slots i in ``0..K``.
Every slot is readable; slots beyond ``BINS`` contain zero. Use ``Striped``
Store and an output ownership grain of K to write increasing bin order.
The complete output capacity ``B * K`` must cover ``BINS``. The bin count,
input tile size and output capacity must fit signed 32-bit indexing.

``Atomic`` counts with atomic additions. ``Sort`` counts by sorting a private
copy; it also preserves the borrowed input. Both support complete
one-dimensional Blocks with positive input and output extents. Every thread
participates with the same static selections. Generic helpers need no
``generic_const_exprs`` feature.

Temporary storage
-----------------

Scratch includes CUB's temporary storage and the intermediate counters for
both algorithms. The operation synchronizes before reading these counters.
Omitted storage uses managed synchronization before later scratch reuse.

Use ``temp_storage(&mut scratch)`` to borrow an explicit descriptor. Manual
reuse requires ``cuda_device::thread::sync_threads()`` after the call;
``auto_sync()`` supplies a trailing barrier. Exclusive storage retains its
existing allocation policy. Unrelated global accesses that require Block
completion still need their own synchronization.

See :doc:`payload`, :doc:`groups` and :doc:`load_store` for checked access,
participation and storage. Generate the API reference with
``cargo doc --workspace --no-deps`` from ``rust/`` for the full signatures.
