.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Merge-sorting numeric keys
===========================

``merge_sort_keys`` sorts a group's keys into a fresh blocked payload:

.. code-block:: rust

   let sorted = coop::merge_sort_keys(block, &keys, Default::default());

The default sorts every key in ascending order. The result retains the input's
numeric type, item count and minimum alignment. Input values and validity remain
unchanged. Merge Sort supports all ten ``Numeric`` types, including ``f32`` and
``f64``. Equal keys have no stable-tie guarantee; floating keys must satisfy the
comparison's strict weak ordering requirements. No NaN total order is supplied.

A complete program
------------------

This helper keeps the item count and alignment generic. It returns a full
ascending result, a descending prefix and the original payload. The partial
operation takes a group-uniform signed count and a sentinel of the key type:

.. literalinclude:: ../../rust/cuda-coop/examples/merge_sort_keys/helper/src/lib.rs
   :language: rust
   :start-after: // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

The host instantiates one and four items per thread on two 4-by-4-by-4 Blocks.
It uses ``i64`` keys with at least 32-byte alignment, checks both sorted results,
verifies input preservation and result validity, and mutates one result while
retaining the other:

.. literalinclude:: ../../rust/cuda-coop/examples/merge_sort_keys/src/main.rs
   :language: rust
   :start-after: // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

With the matching Oxide and ``1.99.1`` installed, run from a complete
CCCL checkout:

.. code-block:: shell

   cd rust/cuda-coop/examples/merge_sort_keys
   CUDA_HOME=/usr/local/cuda-13.3 CUDA_PATH=/usr/local/cuda-13.3 \
     cargo oxide run --arch sm_120 --materialize-cubin

Choose the architecture for your GPU. ``auto_shared`` uses the compiler's
complete temporary-storage plan.

Counts and result validity
--------------------------

``MergeSortOptions::default().valid_items(count, sentinel)`` sorts a blocked
prefix. Count and sentinel are selected together and must be uniform across the
group. The sentinel must compare after every valid key in the selected order:
for example, ``i64::MAX`` for ascending or ``i64::MIN`` for descending keys.
``order(coop::Descending)`` preserves both controls, and the builders can be
applied in either order.

Count is ``i64`` and must be in ``0..=group_size * ITEMS_PER_THREAD``. It is
checked before narrowing to CUB's integer type. Every input slot must be
initialized, including slots outside the prefix and all slots at count zero.
Invalid counts or unreadable inputs trap before native numeric reads.

Only the result's blocked prefix is initialized. ``get`` returns ``None`` for
suffix slots, and Store can write the valid prefix without touching the rest
of the destination. Count zero still participates and returns an empty result.

Groups and temporary storage
----------------------------

Every member of the selected group participates. Complete XYZ Blocks require a
power-of-two thread count. Physical Warps and logical widths 1, 2, 4, 8, 16 and
32 are supported with complete physical warps in the enclosing Block; that
Block's total need not be a power of two. Per-thread item counts are positive.
Input and result slots use blocked order within each group.

Implicit storage is compiler-planned for each participating group. Blocks can
also borrow existing ``TempStorage`` through ``temp_storage(&mut scratch)``.
Manual storage requires Block completion before conflicting reuse; auto-sync
storage supplies its trailing barrier. Explicit storage is not a Warp option.
Internal CUB synchronization does not complete unrelated global-memory accesses.

See :doc:`payload`, :doc:`groups` and :doc:`load_store` for payload initialization,
participation and storage reuse. :doc:`radix_sort_keys` provides stable ordering
of selected integer-key digits.
