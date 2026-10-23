.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Merge-sorting key-value pairs
===============================

``merge_sort_pairs`` sorts keys and carries each associated value into two fresh
blocked payloads:

.. code-block:: rust

   let (sorted_keys, sorted_values) =
       coop::merge_sort_pairs(block, &keys, &values, Default::default());

Keys and values independently support the ten ``Numeric`` types. They have the
same per-thread item count and may have different minimum alignments. Each
result retains its corresponding input's type and alignment. Both inputs and
their validity remain unchanged. The default sorts the full group in ascending
key order with implicit temporary storage.

Equal keys have no stable-tie guarantee. Values stay associated with their keys,
including floating-point value bit patterns. Floating keys must satisfy the
comparison's strict weak ordering requirements; no NaN total order is supplied.
Use :doc:`radix_sort_pairs` for stable selected-digit sorting of integer keys.

A complete program
------------------

This helper preserves generic item counts and independent key/value alignments.
It returns a full ascending pair, a descending prefix pair and both inputs:

.. literalinclude:: ../../rust/cuda-coop/examples/merge_sort_pairs/helper/src/lib.rs
   :language: rust
   :start-after: // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

The host instantiates one and four items per thread on two 4-by-4-by-4 Blocks.
It uses ``i32`` keys with at least 16-byte alignment and ``f64`` values with at
least 32-byte alignment. Its checks use key ordering and pair-bit multisets so
equal-key permutations are allowed, while association, preserved inputs, prefix
validity and fresh result mutations are observed independently:

.. literalinclude:: ../../rust/cuda-coop/examples/merge_sort_pairs/src/main.rs
   :language: rust
   :start-after: // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

With the pinned Oxide revision and Rust ``1.99.1``
installed, run from a complete CCCL checkout:

.. code-block:: shell

   cd rust/cuda-coop/examples/merge_sort_pairs
   CUDA_HOME=/usr/local/cuda-13.3 CUDA_PATH=/usr/local/cuda-13.3 \
     cargo oxide run --arch sm_120 --materialize-cubin

Choose the architecture for your GPU. ``auto_shared`` uses the compiler's
complete temporary-storage plan.

Counts, validity and storage
----------------------------

``MergeSortOptions::default().valid_items(count, sentinel)`` selects a blocked
prefix of pairs. Count is a group-uniform ``i64`` in
``0..=group_size * ITEMS_PER_THREAD``. The sentinel has the **key** type and must
compare after every valid key in the chosen order. There is no value sentinel.
The count and sentinel are selected together; ``order`` and ``temp_storage``
retain them regardless of builder order.

Every slot in both input payloads must be initialized, even outside the prefix
and when count is zero. Native preflight checks both validity maps before
reading any numeric input. Invalid counts or unreadable inputs trap. Both fresh
results have exactly the prefix initialized; ``get`` returns ``None`` for either
suffix, and a prefix Store leaves destination suffix slots untouched.

Complete XYZ Blocks require a power-of-two thread count. Physical Warps and
logical widths 1, 2, 4, 8, 16 and 32 use complete physical warps in the enclosing
Block. Implicit storage is planned separately for participating groups. Blocks
can borrow existing ``TempStorage`` with ``temp_storage(&mut scratch)``; manual
reuse requires Block completion, while auto-sync storage supplies its trailing
barrier. Explicit storage is not a Warp option. Both result payloads remain
independent of scratch reuse. Internal CUB synchronization does not complete
unrelated global-memory accesses.

See :doc:`merge_sort_keys`, :doc:`payload`, :doc:`groups` and :doc:`load_store`
for shared ordering options, initialization, participation and storage rules.
