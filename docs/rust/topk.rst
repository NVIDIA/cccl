.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Selecting TopK keys and pairs
================================

This example requires Oxide's generic native-registration support. The revision
currently pinned by the Cargo manifests does not yet include that support.

``topk_min_keys`` and ``topk_max_keys`` return the smallest or largest keys in a
fresh, unsorted payload. ``topk_min_pairs`` and ``topk_max_pairs`` also carry each
associated value into an independently typed fresh payload:

.. code-block:: rust

   let (selected_keys, selected_values) = coop::topk_min_pairs(
       block, &keys, &values, coop::TopKOptions::default().k_const::<17>(),
   );

All four operations require a selection count, ``k``. There is no default count
or sorting option. Keys and values independently support the ten ``Numeric``
types, share the per-thread item count and retain their own minimum alignments.
Both input payloads and their validity remain unchanged.

A complete program
------------------

The helper keeps the item count and both alignments generic. It exercises all
four operations with static and runtime counts, returning the selections and
both input payloads:

.. literalinclude:: ../../rust/cuda-coop/examples/topk/helper/src/lib.rs
   :language: rust
   :start-after: // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

The host instantiates one and four items per thread on two 64-thread Blocks.
It uses ``i32`` keys with at least 16-byte alignment and ``f64`` values with at
least 32-byte alignment. The checks observe selection, exact key/value bit
association, preserved inputs, prefix validity and independent result mutation:

.. literalinclude:: ../../rust/cuda-coop/examples/topk/src/main.rs
   :language: rust
   :start-after: // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

With an Oxide revision containing that support and ``nightly-2026-10-11``
installed, run from a complete CCCL checkout:

.. code-block:: shell

   cd rust/cuda-coop/examples/topk
   CUDA_HOME=/usr/local/cuda-13.3 CUDA_PATH=/usr/local/cuda-13.3 \
     cargo oxide run --arch sm_120 --materialize-cubin

Choose the architecture for your GPU. ``auto_shared`` uses the compiler's
complete temporary-storage plan.

Counts and result validity
--------------------------

``TopKOptions::default().k(count)`` selects a uniform signed64 runtime count;
``k_const::<COUNT>()`` selects a static count. Omitting ``valid_items`` selects
the complete input tile. ``valid_items(count)`` or
``valid_items_const::<COUNT>()`` selects a valid input prefix. These builders
retain each other's selections in either order. An explicit full-tile count
still uses the partial-input form.

Both counts must lie in ``0..=block_size * ITEMS_PER_THREAD``. The static count
is checked when the concrete specialization is compiled. Runtime counts are
checked at their full signed64 width before narrowing. If k exceeds the valid
count, all valid inputs are selected. Each output initializes exactly the first
``min(k, valid_items)`` slots in blocked order. Its suffix remains invalid:
``get`` returns ``None``, and Store must be limited to the selected prefix.
A prefix Store leaves the destination suffix untouched.

Every slot in every input payload must be initialized, even outside the valid
prefix and even when either count is zero. Native preflight checks both maps
for a pair operation before reading numeric inputs. Invalid counts or
unreadable input trap. A uniform empty selection skips CUB after validation,
while retaining collective participation and storage obligations.

Selected outputs are **not sorted**. Equal-key choice and order are unspecified;
positive and negative zero compare as equal, and no NaN key ordering is
promised. Associated values move without numeric conversion, preserving their
bits, including NaNs and signed zeros. Use :doc:`radix_sort_pairs` when stable
selected-digit sorting of integer keys is required.

Groups and temporary storage
----------------------------

TopK supports complete one-dimensional Blocks of 1 through 1024 threads.
The tile size must fit CUB's signed int capacity. Warp and multidimensional
Block groups are not TopK options.

Implicit scratch uses the existing compiler planner. A Block can borrow
``TempStorage`` with ``temp_storage(&mut scratch)``. Manual storage requires
Block completion before conflicting reuse, including after an empty selection.
Auto-sync storage supplies its trailing barrier. Both result payloads remain
independent of scratch reuse. Internal CUB synchronization does not complete
unrelated global-memory accesses.

See :doc:`payload`, :doc:`groups` and :doc:`load_store` for initialization,
participation and storage rules.
