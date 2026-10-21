.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Shift a block's values
=======================

``shuffle`` shifts a block's values by one item in blocked order. It preserves
the input and returns a fresh ``ThreadData`` with the same element type, item
count and alignment. The default direction, ``Down``, reads the next item;
``Up`` reads the previous item:

.. code-block:: rust

   let next = coop::shuffle(block, &values, Default::default());
   let previous = coop::shuffle(block, &values,
       coop::ShuffleOptions::default().mode(coop::Up));

The exposed edge remains invalid. For a block containing ``[10, 20, 30, 40]``
in blocked order, Down produces ``[20, 30, 40, invalid]`` and Up produces
``[invalid, 10, 20, 30]``. Values cross thread boundaries; the shift neither
wraps around the block nor supplies a boundary value.

A complete program
------------------

This helper loads a complete tile, shifts it, then initializes the exposed
edge through the safe payload API before storing. The input determines the
element type. ``ITEMS_PER_THREAD`` and the direction remain generic:

.. literalinclude:: ../../rust/cuda-coop/examples/shuffle/helper/src/lib.rs
   :language: rust
   :start-after: // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

The host program specializes both directions for one and four items per
thread. Each of its three blocks shifts its own tile independently, with
zero supplied explicitly by the helper's caller for the missing edge:

.. literalinclude:: ../../rust/cuda-coop/examples/shuffle/src/main.rs
   :language: rust
   :start-after: // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Use the Oxide revision selected by the manifests and ``nightly-2026-10-11``.
With that toolchain installed, run from a complete CCCL checkout:

.. code-block:: shell

   cd rust/cuda-coop/examples/shuffle
   CUDA_HOME=/usr/local/cuda-13.3 CUDA_PATH=/usr/local/cuda-13.3 \
     cargo oxide run --arch sm_120 --materialize-cubin

Choose the architecture for your GPU. The compiler supplies the shared-storage
requirement to ``auto_shared``; the consumer does not calculate scratch bytes.

Inputs and result validity
----------------------------

Shuffle supports complete one-, two- and three-dimensional Blocks, including
sizes that are not multiples of 32. Thread rank is x-fastest. Every thread
must participate with the same direction, element type and positive item count.
All ten ``Numeric`` types are supported. Ordinary same-extent helpers need
no ``generic_const_exprs`` feature.

Every input slot must be initialized, including the one whose value does not
contribute to the result. A masked Load without a default can leave invalid
slots; Shuffle traps before reading them. The input remains unchanged.

Down leaves the last item of the last thread invalid. Up leaves the first
item of the first thread invalid. ``get`` returns ``None`` at that edge, and
checked indexing or a Store that needs it traps. Use ``write`` to initialize
the slot before reading or storing it. A one-thread block with one item
returns an entirely invalid payload.

``ShuffleOptions::default()`` selects Down. ``mode(Up)`` selects Up, and the
optional ``distance::<1>()`` spells the fixed distance explicitly. Other
distances fail compilation. The operation accepts payloads and complete
Blocks; it has no scalar, Warp, fill, count, algorithm or explicit-storage form.

Participation and storage
-------------------------

Oxide checks participation and plans the actual CUB shared storage, including
synchronization before reuse. Keep collective calls outside lane-specific
branches and reconverge after repairing the edge, as the helper does.
Shuffle does not promise completion of earlier global-memory accesses. Use
``cuda_device::thread::sync_threads()`` when dependent accesses need block completion.

See :doc:`payload` for checked access and :doc:`groups` for participation.
Generate the item reference with ``cargo doc --workspace --no-deps`` from
``rust/``; ``shuffle``, ``ShuffleOptions`` and ``ShuffleMode`` document the
full signatures and selectors.
