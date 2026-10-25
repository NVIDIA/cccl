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

Stateless comparators
---------------------

Full-input Block ``i32`` Keys also accepts function items and zero-capture
``Fn(i32, i32) -> bool + Copy`` closures through ``order``:

.. code-block:: rust

   let descending = coop::merge_sort_keys(
       block, &keys,
       coop::MergeSortOptions::default().order(|left, right| left > right),
   );

True means that the left key precedes the right. The comparator must define a
strict weak ordering; ties remain unordered. Callable sorting keeps the same
fresh result, input validity and temporary-storage rules. Captures, function
pointers, partial tiles, Warp groups and Pairs are rejected at compile time for
callable orders. Closures that call methods on their keys may need explicit
``i32`` parameter types.
Built-in ascending and descending orders retain their existing type and group
support.

A complete program
------------------

This helper keeps the item count and alignment generic. It returns a full
ascending result, a descending prefix and the original payload. The partial
operation takes a group-uniform signed count and a sentinel of the key type.
The second helper forwards ``MergeSortOptions`` under the existing nongeneric
``MergeSortOrder`` bound, preserving the actual comparator and selected storage:

.. literalinclude:: ../../rust/cuda-coop/examples/merge_sort_keys/helper/src/lib.rs
   :language: rust
   :start-after: // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

The host instantiates one and four items per thread on two 4-by-4-by-4 Blocks.
It uses ``i64`` keys with at least 32-byte alignment, checks both sorted results,
verifies input preservation and result validity, and mutates one result while
retaining the other. A second kernel uses three 128-thread Blocks with ``i32``
keys, including duplicates and both integer extremes. For one and four items
per thread it combines built-in ascending, an even-first function item and a
descending closure. Ascending and even-first reuse auto-sync scratch; the
descending closure uses implicit storage. All three results remain live, and
the host checks every result, validity bit and preserved input:

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
