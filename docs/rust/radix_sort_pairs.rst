.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Sorting key/value pairs
=======================

``radix_sort_pairs`` stably sorts a complete Block's integer keys and carries
each associated value with its key. It borrows two payloads and returns two
fresh payloads:

.. code-block:: rust

   let (sorted_keys, sorted_values) = coop::radix_sort_pairs(
       block, &keys, &values, Default::default());

The default sorts all key bits in ascending order. Keys support ``i32``, ``u32``,
``i64`` and ``u64``. Values support ``i8``, ``u8``, ``i16``, ``u16``, ``i32``,
``u32``, ``i64``, ``u64``, ``f32`` and ``f64``. Values are moved without
comparisons or arithmetic; floating-point values retain their bit patterns.

Both inputs remain unchanged and must have the same positive item count per
thread. Their minimum alignments are independent. Each output retains its
input's type, item count and alignment; changing a result does not change either
input or another result. Every slot in both inputs must be initialized. Both
validity maps are checked before native sorting reads either numeric array.

A complete program
------------------

This separate helper infers key and value types from Load and forwards a generic
item count and independent alignments. It returns ascending and descending sorts
of selected digits together with both original payloads:

.. literalinclude:: ../../rust/cuda-coop/examples/radix_sort_pairs/helper/src/lib.rs
   :language: rust
   :start-after: // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

The host instantiates one and four items per thread on 4-by-4-by-4 Blocks, with
``i32`` keys and ``f64`` values aligned to at least 16 and 32 bytes respectively.
Selected digits include the signed-key sign bit. Distinct lower-bit tags and
associated values distinguish stable selected-digit ordering from sorting whole
keys. The program checks both sorts, both preserved inputs and mutations of
fresh results; floating values are compared by their bits:

.. literalinclude:: ../../rust/cuda-coop/examples/radix_sort_pairs/src/main.rs
   :language: rust
   :start-after: // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

With the pinned Oxide revision and Rust ``1.99.1``
installed, run from a complete CCCL checkout:

.. code-block:: shell

   cd rust/cuda-coop/examples/radix_sort_pairs
   CUDA_HOME=/usr/local/cuda-13.3 CUDA_PATH=/usr/local/cuda-13.3 \
     cargo oxide run --arch sm_120 --materialize-cubin

Choose the architecture for your GPU. ``auto_shared`` uses the compiler's
complete temporary-storage plan for the paired primitive.

Order, bounds and storage
-------------------------

``RadixSortOptions`` is shared with :doc:`radix_sort_keys`. The default interval
is ``[0, key_width)``. ``begin_bit`` and ``end_bit`` independently replace an
endpoint; an omitted end remains the key width even with a nonzero begin.
``order(coop::Descending)`` selects descending order. Both directions preserve
the original order of pairs whose selected key digits are equal. Signed keys
invert the sign bit before selecting digits and restore their original
representation in the result.

Value builders accept Block-uniform ``i64`` endpoints. The native operation
checks ``0 <= begin < end <= key_width`` before narrowing or reading inputs.
Invalid runtime intervals, including empty ones, trap. Const builders
``begin_bit_const::<B>()`` and ``end_bit_const::<E>()`` request checks when the
operation is instantiated. Mixed static and runtime bounds are supported.

Every member of the complete XYZ Block participates. Input and output use
blocked order: x-fastest thread rank r owns items ``r * ITEMS_PER_THREAD + i``.
The complete tile contains at most 65,535 pairs. There is no partial-input or
fill option.

Omitted scratch uses compiler-planned CUB storage with synchronization before
dirty reuse. ``temp_storage(&mut scratch)`` retains the descriptor's policy:
manual reuse requires ``cuda_device::thread::sync_threads()`` after the call,
while ``auto_sync()`` supplies that completion. Internal sorting barriers do not
complete unrelated global-memory accesses. See :doc:`payload`, :doc:`groups`
and :doc:`load_store` for initialization, participation and checked access.
