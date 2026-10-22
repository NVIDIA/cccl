.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Flags at changes between values
===============================

``discontinuity`` marks changes between neighboring items in blocked order.
It preserves the input and returns fresh ``i32`` flags: one for unequal
neighbors, zero for equal neighbors. ``Heads`` is the default and compares
each item with its predecessor. ``Tails`` compares with its successor.
``HeadsAndTails`` returns two independent payloads, heads first:

.. code-block:: rust

   let heads = coop::discontinuity(block, &values, Default::default());
   let (heads, tails) = coop::discontinuity(block, &values,
       coop::DiscontinuityOptions::default().mode(coop::HeadsAndTails));

For ``[2, 2, 5, 5]``, heads are ``[1, 0, 1, 0]`` and tails are
``[0, 1, 0, 1]``. Neighbors cross thread boundaries. Without an outside
neighbor, the first head and last tail are one.

A complete program
------------------

The helper infers the input type and keeps ``ITEMS_PER_THREAD`` generic.
It loads a complete tile and returns both flag payloads and the preserved
input. The flag type is ``i32`` even when the input is ``f64``:

.. literalinclude:: ../../rust/cuda-coop/examples/discontinuity/helper/src/lib.rs
   :language: rust
   :start-after: // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

The host program uses a ``(4, 3, 2)`` block, specializes one and four items
per thread, and checks both flag arrays and the preserved input:

.. literalinclude:: ../../rust/cuda-coop/examples/discontinuity/src/main.rs
   :language: rust
   :start-after: // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Use the Oxide revision selected by the manifests and ``nightly-2026-10-11``.
With that toolchain installed, run from a complete CCCL checkout:

.. code-block:: shell

   cd rust/cuda-coop/examples/discontinuity
   CUDA_HOME=/usr/local/cuda-13.3 CUDA_PATH=/usr/local/cuda-13.3 \
     cargo oxide run --arch sm_120 --materialize-cubin

Choose the architecture for your GPU. ``auto_shared`` uses the compiler's
plan for the actual CUB temporary-storage requirement.

Outside neighbors and initialized input
---------------------------------------

Heads accepts ``tile_predecessor_item(value)``; Tails accepts
``tile_successor_item(value)``. The combined mode accepts either, both or
neither. Outside neighbors have exactly the input type and are uniform
across the block. A supplied neighbor makes its edge flag a normal
comparison: predecessor two changes the example's first head to zero;
successor five changes its last tail to zero. Builders preserve both fields
when changing mode.

Every input slot must be initialized. There is no count option: all values,
including any fill supplied to Load, participate in comparisons. Floating
inequality treats NaNs as unequal and positive and negative zero as equal.
Every output slot is initialized. Both outputs retain the input's item
count and minimum alignment, with the natural alignment required by ``i32``.
Changing either result cannot change the input or the other result.

Groups and temporary storage
----------------------------

Complete one-, two- and three-dimensional Blocks and all ten ``Numeric``
input types are supported. Rank is x-fastest, item count is positive, and
the complete tile must fit CUB's signed 32-bit indexing range. Every thread
participates with the same mode, type and extent. Generic helpers can return
one payload or the pair without a ``generic_const_exprs`` feature.

Omitted storage uses managed synchronization before scratch reuse. Use
``temp_storage(&mut scratch)`` for an explicit descriptor: manual reuse
requires block completion, ``auto_sync()`` supplies a trailing barrier,
and exclusive storage keeps its existing allocation policy. For unrelated
global accesses that need block completion, use
``cuda_device::thread::sync_threads()``.

See :doc:`payload`, :doc:`groups` and :doc:`load_store` for checked access,
participation and storage. Generate the API reference with
``cargo doc --workspace --no-deps`` from ``rust/`` for the full signatures
and mode-dependent result types.
