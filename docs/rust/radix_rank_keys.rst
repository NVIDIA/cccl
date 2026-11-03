.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Ranking integer key digits
==========================

``radix_rank_keys`` assigns each input item its zero-based position in the
stable order of selected key digits across a complete Block:

.. code-block:: rust

   let ranks = coop::radix_rank_keys(block, &keys, Default::default());

The default ranks the low four bits in ascending order. Each rank stays with
its original input item; equal digits receive increasing ranks in original
blocked order. The fresh result is ``ThreadData<i32, ITEMS_PER_THREAD>`` with
every slot initialized and natural ``i32`` alignment, independently of the key
payload's alignment. Input values and validity remain unchanged.

A complete program
------------------

This helper infers its key type from Load and forwards a generic item count and
alignment. It returns ascending and descending ranks plus the original payload.
The ascending interval derives its width from static endpoints; the descending
interval supplies an explicit width with a runtime begin:

.. literalinclude:: ../../rust/cuda-coop/examples/radix_rank_keys/helper/src/lib.rs
   :language: rust
   :start-after: // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

The host instantiates one and four items per thread on two 4-by-4-by-4 Blocks.
It uses ``i64`` keys aligned to at least 32 bytes and selects the three high bits,
including the signed-key sign bit. Distinct lower-bit tags distinguish digit
ranking from whole-key ordering. Independent host sorts check stable ties by
inverting each sorted permutation, then compare both rank payloads and verify
the preserved payload and source keys:

.. literalinclude:: ../../rust/cuda-coop/examples/radix_rank_keys/src/main.rs
   :language: rust
   :start-after: // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

With the pinned Oxide revision and Rust ``1.99.1``
installed, run from a complete CCCL checkout:

.. code-block:: shell

   cd rust/cuda-coop/examples/radix_rank_keys
   CUDA_HOME=/usr/local/cuda-13.3 CUDA_PATH=/usr/local/cuda-13.3 \
     cargo oxide run --arch sm_120 --materialize-cubin

Choose the architecture for your GPU. ``auto_shared`` uses the compiler's
complete temporary-storage plan.

Digits and participation
------------------------

Keys support ``i32``, ``u32``, ``i64`` and ``u64``. Signed keys invert their sign
bit before digit extraction. ``RadixRankOptions`` defaults to ascending order,
begin zero and width four. ``order(coop::Descending)`` selects descending order
while preserving tie order.

The static radix width must be in ``1..=8``. ``radix_bits::<W>()`` selects it
explicitly. With a static end and no explicit width, the width is
``end - begin``; use ``begin_bit_const::<B>()`` and ``end_bit_const::<E>()`` to
select static endpoints. A Block-uniform ``begin_bit(i64)`` requires an explicit
width, even when passed a literal. An optional end is always static and must
equal begin plus width. Builders preserve other selections and validate the
final combined choices.

Static choices are checked when the operation is instantiated. Runtime bounds
must satisfy ``0 <= begin`` and ``begin + width <= key_width``, together with
any selected end. Invalid bounds and unreadable input slots trap before numeric
reads. Every input slot must be initialized.

Every member of the complete XYZ Block participates. Positive per-thread item
counts are supported; the tile contains at most 65,535 keys. Input and result
slots use blocked order: x-fastest thread rank r owns items
``r * ITEMS_PER_THREAD + i``.

Rank uses implicit compiler-planned CUB scratch with synchronization before
dirty reuse. Internal barriers do not complete unrelated global-memory
accesses. See :doc:`payload` and :doc:`groups` for initialization and
participation, or :doc:`radix_sort_keys` to return keys in sorted order.
