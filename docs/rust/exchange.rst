.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Exchange layouts
================

``exchange`` redistributes a group's per-thread values between striped and
blocked layouts. It preserves the input and returns a fresh ``ThreadData``
with the same element type, item count and alignment. The default converts
striped input to blocked output:

.. code-block:: rust

   let blocked = coop::exchange(group, &striped, Default::default());
   let striped_again = coop::exchange(group, &blocked,
       coop::ExchangeOptions::default().mode(coop::BlockedToStriped));

Blocked order assigns consecutive items to each thread. Striped order assigns
consecutive items to consecutive threads. For four threads with two items each,
``StripedToBlocked`` produces:

.. list-table::
   :header-rows: 1

   * - Thread rank
     - Striped input
     - Blocked output
   * - 0
     - ``[0, 4]``
     - ``[0, 1]``
   * - 1
     - ``[1, 5]``
     - ``[2, 3]``
   * - 2
     - ``[2, 6]``
     - ``[4, 5]``
   * - 3
     - ``[3, 7]``
     - ``[6, 7]``

``BlockedToStriped`` reverses this conversion. Exchange operates on the values
already held by the threads; Load and Store handle global-memory access.

A complete program
------------------

This standalone Cargo example loads a tile in striped order, exchanges it to
blocked order, then uses Direct Store. Its reusable helper lives in a separate
crate. The element type is inferred, and ``ITEMS_PER_THREAD`` stays generic
through the host launch, kernel and helper. These same-extent signatures need
no ``generic_const_exprs`` feature in the consumer.

.. literalinclude:: ../../rust/cuda-coop/examples/exchange/helper/src/lib.rs
   :language: rust
   :start-after: // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

The host specializes one and four items per thread across three blocks. A
filled Load initializes the short final tile, and Store writes only its valid
prefix. The resulting global output matches the input even though the
per-thread layout changes between Load and Store.

.. literalinclude:: ../../rust/cuda-coop/examples/exchange/src/main.rs
   :language: rust
   :start-after: // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Use the Oxide revision selected by the manifests and ``nightly-2026-10-11``.
With that toolchain installed, run from a complete CCCL checkout:

.. code-block:: shell

   cd rust/cuda-coop/examples/exchange
   CUDA_HOME=/usr/local/cuda-13.3 CUDA_PATH=/usr/local/cuda-13.3 \
     cargo oxide run --arch sm_120 --materialize-cubin

Choose the architecture for your GPU. ``auto_shared`` uses the compiler's
inferred storage plan; the consumer does not calculate scratch bytes.

Supported groups and inputs
---------------------------

Exchange supports complete one-, two- and three-dimensional Blocks, physical
Warps and logical Warp groups of widths 1, 2, 4, 8, 16 and 32. Block thread rank
is x-fastest. Block size need not be a multiple of 32; Warp operations require
complete physical warps in the enclosing Block. Each Warp group converts its
own values independently.

The payload has a positive item count and one of the ten ``Numeric`` element
types: ``i8``, ``u8``, ``i16``, ``u16``, ``i32``, ``u32``, ``i64``, ``u64``,
``f32`` or ``f64``. Pass a shared borrow, or an ordinary mutable reborrow.
Every input slot must be initialized, including a one-item payload. A masked
Load without a default can leave unwritten slots; Exchange traps before
reading them. Every returned slot is initialized, and changing the result
does not change the input.

``ExchangeOptions::default()`` selects ``StripedToBlocked``. The consuming
``mode`` builder selects either direction at compile time. The API supports
these two payload conversions; scalar inputs, ranked scatter, algorithm and
explicit-storage options are unsupported.

Participation and storage
-------------------------

Every selected member must reach the same Exchange. Separate Warp groups may
take different paths, but members of one group must participate together.
Oxide checks participation and plans the actual CUB shared storage, including
independent regions for Warp groups and synchronization before reuse.

Exchange does not promise completion of earlier global-memory accesses. Use
the group's ``sync()`` when subsequent conflicting accesses need completion.
See :doc:`groups` and :doc:`load_store` for the participation and memory rules.

Generate the item reference with ``cargo doc --workspace --no-deps`` from
``rust/``. ``exchange``, ``ExchangeOptions``, ``ExchangeGroup`` and
``ExchangeMode`` document the full generic signatures and supported selectors.
