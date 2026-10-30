.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Sorting integer keys
====================

``radix_sort_keys`` stably sorts a complete Block's integer keys and returns a
fresh payload with the same type, item count and alignment. Default options
sort all bits in ascending order:

.. code-block:: rust

   let sorted = coop::radix_sort_keys(block, &keys, Default::default());

The input remains unchanged. Every input slot must be initialized; unreadable
slots trap before native sorting reads. Results use blocked order: thread rank
r owns positions ``r * ITEMS_PER_THREAD + i``. Rank is x-fastest across the
Block's three dimensions.

A complete program
------------------

This helper infers its key type from Load and forwards generic item count and
alignment. It returns a full-width sort, two selected-bit sorts and the original
payload:

.. literalinclude:: ../../rust/cuda-coop/examples/radix_sort_keys/helper/src/lib.rs
   :language: rust
   :start-after: // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

The host instantiates one and four items per thread on 4-by-4-by-4 Blocks.
Signed extrema and keys sharing selected digits exercise ordering and stable
ties. Independent host sorts check all results and input preservation:

.. literalinclude:: ../../rust/cuda-coop/examples/radix_sort_keys/src/main.rs
   :language: rust
   :start-after: // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

With the pinned Oxide revision and Rust ``1.99.1``
installed, run from a complete CCCL checkout:

.. code-block:: shell

   cd rust/cuda-coop/examples/radix_sort_keys
   CUDA_HOME=/usr/local/cuda-13.3 CUDA_PATH=/usr/local/cuda-13.3 \
     cargo oxide run --arch sm_120 --materialize-cubin

Choose the architecture for your GPU. ``auto_shared`` uses the compiler's
complete temporary-storage plan.

Selecting bits and order
------------------------

Keys support ``i32``, ``u32``, ``i64`` and ``u64``. The default interval is
``[0, key_width)``. ``begin_bit`` and ``end_bit`` independently replace an
endpoint; an omitted end always remains the full key width. Use
``order(coop::Descending)`` for descending order. Both directions retain the
original blocked order of keys whose selected digits are equal.

Signed keys invert their sign bit before selecting the interval. The result
contains the original key representation, including the bits outside that
interval.

Value builders accept Block-uniform ``i64`` endpoints. The native operation
checks ``0 <= begin < end <= key_width`` before narrowing them or reading keys.
Empty intervals and out-of-range runtime endpoints trap. A literal passed to a
value builder follows this runtime route. To request compile-time validation,
use ``begin_bit_const::<B>()`` or ``end_bit_const::<E>()``:

.. code-block:: rust

   let selected = coop::radix_sort_keys(block, &keys,
       coop::RadixSortOptions::default()
           .begin_bit_const::<8>().end_bit(16).order(coop::Descending));

Static endpoints are checked when the operation is instantiated. Mixed static
and runtime endpoints are supported. Builders preserve other selections and
validate the final options, so replacing an endpoint uses its final value.

Participation and storage
-------------------------

Every thread in a complete Block participates. Positive per-thread extents and
one-, two- or three-dimensional Blocks are supported; the complete tile must
contain at most 65,535 keys. This operation has no partial-count or fill option.

Omitted storage uses compiler-planned CUB scratch with synchronization before
dirty reuse. ``temp_storage(&mut scratch)`` borrows an explicit descriptor;
manual reuse requires ``cuda_device::thread::sync_threads()`` after the call,
and ``auto_sync()`` supplies that trailing barrier. Internal CUB barriers do
not complete unrelated global-memory accesses.

See :doc:`payload`, :doc:`groups` and :doc:`load_store` for checked access,
participation and storage. Generate the full API reference with
``cargo doc --workspace --no-deps`` from ``rust/``.
