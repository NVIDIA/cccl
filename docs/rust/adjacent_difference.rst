.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Differences between neighboring values
=======================================

``adjacent_difference`` subtracts neighboring values in a block's blocked
order. It preserves the input and returns a fresh ``ThreadData`` with the
same element type, item count and alignment. The default ``Left`` direction
subtracts the previous item; ``Right`` subtracts the next item:

.. code-block:: rust

   let left = coop::adjacent_difference(block, &values, Default::default());
   let right = coop::adjacent_difference(block, &values,
       coop::AdjacentDifferenceOptions::default().direction(coop::Right));

For ``[10, 20, 40, 70]``, Left produces ``[10, 10, 20, 30]`` and Right
produces ``[-10, -20, -30, 70]``. Neighbors cross thread boundaries. Without
an outside neighbor, the first Left item or last Right item is copied.

A complete program
------------------

The helper keeps ``ITEMS_PER_THREAD`` generic and infers the element type
from the input slice. Load initializes the payload, Adjacent Difference
returns fresh values, and Store writes the result:

.. literalinclude:: ../../rust/cuda-coop/examples/adjacent_difference/helper/src/lib.rs
   :language: rust
   :start-after: // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

The host program specializes one and four items per thread:

.. literalinclude:: ../../rust/cuda-coop/examples/adjacent_difference/src/main.rs
   :language: rust
   :start-after: // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Use the Oxide revision selected by the manifests and ``nightly-2026-10-11``.
With that toolchain installed, run from a complete CCCL checkout:

.. code-block:: shell

   cd rust/cuda-coop/examples/adjacent_difference
   CUDA_HOME=/usr/local/cuda-13.3 CUDA_PATH=/usr/local/cuda-13.3 \
     cargo oxide run --arch sm_120 --materialize-cubin

Choose the architecture for your GPU. ``auto_shared`` uses the compiler's
shared-storage plan, including the actual CUB temporary-storage requirement.

Counts and outside neighbors
-----------------------------

``valid_items(count)`` limits the subtraction to the first ``count`` items
of the whole block tile. Every later item is copied unchanged. With count
two, the example above produces ``[10, 10, 40, 70]`` for Left and
``[-10, 20, 40, 70]`` for Right. Count zero returns a copy of the whole input.
Counts must be block-uniform and lie between zero and the tile capacity.

Left accepts ``tile_predecessor_item(value)`` with or without a count.
Right accepts ``tile_successor_item(value)`` only when the count is omitted.
A supplied count and successor are rejected even when the count equals the
full tile size. Outside neighbors have exactly the input element type and
must be block-uniform. Changing direction preserves the option fields;
it does not reinterpret a predecessor as a successor.

Every input slot must be initialized, including the copied suffix and every
slot at count zero. Fill missing Load items or initialize them with ``write``
before calling. Every result slot is initialized; changing a result cannot
change the input. Arithmetic uses native subtraction in the input type.

Groups and temporary storage
----------------------------

The operation accepts complete one-, two- and three-dimensional Blocks and
all ten ``Numeric`` types. Rank is x-fastest; item count is positive and the
whole tile must fit CUB's signed 32-bit indexing range. All threads participate
with the same direction, type and extent. Scalars and Warp groups are not
supported. Same-extent generic helpers need no ``generic_const_exprs`` feature.

Omitted storage uses managed synchronization before scratch reuse. Borrow an
explicit descriptor with ``temp_storage(&mut scratch)`` to use the policies
described in :doc:`load_store`: manual reuse requires block completion,
``auto_sync()`` supplies a trailing barrier, and exclusive storage has its
own allocation. Adjacent Difference does not itself promise completion of
unrelated global-memory accesses; use ``cuda_device::thread::sync_threads()``
when those accesses need block completion.

See :doc:`payload` for checked access and :doc:`groups` for participation.
Generate the API reference with ``cargo doc --workspace --no-deps`` from
``rust/`` for the full signatures and legal option combinations.
