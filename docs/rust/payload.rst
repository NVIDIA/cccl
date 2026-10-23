.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Per-thread payloads
===================

The unpublished ``cuda-coop`` crate provides ``ThreadData`` and its sealed
``ThreadDataLike`` interface. It is a feature-free ``no_std`` crate using the
pinned Oxide device dependency and ``1.99.1`` toolchain. See
:doc:`groups` for physical and mapped group queries. Collective operations and
scratch storage are still under development.

Construct and access a payload
------------------------------

``ThreadData<T, const ITEMS: usize, const ALIGN: usize = 1>`` owns a fixed number
of per-thread slots. ``ITEMS`` must be positive. The element type ``T`` is inferred
from typed writes and preserves its width: the supported types are ``i8``,
``u8``, ``i16``, ``u16``, ``i32``, ``u32``, ``i64``, ``u64``, ``f32`` and ``f64``.

New slots are unwritten. ``write(index, value)`` initializes or replaces exactly
one slot. ``get(index)`` returns a copied value in ``Some`` for initialized slots,
and ``None`` for unwritten or out-of-bounds slots. ``get_mut(index)`` applies the
same checks before returning an exclusive reference.

Indexing reads or mutates an initialized slot. Reading or assigning through an
unwritten index panics; use ``write`` for first initialization. Out-of-bounds
indexing and writes also panic. ``len()`` counts all slots, including unwritten
ones. Copies, moves, swaps and conditional selection preserve the exact validity
of each slot. The payload exposes no unrestricted initialized slice.

This complete host program uses a generic per-thread item count and an inferred
element type:

.. literalinclude:: ../../rust/cuda-coop/examples/payload.rs
   :language: rust
   :start-after: //! Construct and inspect a per-thread payload on the host.

Run it from the Rust workspace:

.. code-block:: shell

   cargo run -p cuda-coop --example payload

Generic payload helpers can use ``D: ThreadDataLike``, with ``D::Item`` for the
element type and ``D::EXTENT`` for the compile-time slot count. The trait's
``get`` and ``write`` methods preserve the same access rules. The trait is sealed
to the crate's supported payload implementations.

Alignment and layout
--------------------

``ALIGN`` is a minimum alignment in bytes for the actual contiguous value
storage. Its default of 1 adds no requirement beyond the element type's natural
alignment. All powers of two from 1 through 2²⁹ are supported. Zero, non-powers of
two and larger values fail to compile; zero item counts fail when the constructor
is instantiated. Large alignments may increase object size through padding.

Code with generic alignment uses the public bound
``Alignment<ALIGN>: SupportedAlignment``. The representation keeps validity
metadata private and does not establish a public foreign-function ABI. Payload
alignment does not allocate scratch memory or establish collective participation
or synchronization.

Validation and API reference
----------------------------

.. code-block:: shell

   cargo test -p cuda-coop
   cargo clippy -p cuda-coop --all-targets -- -D warnings
   RUSTDOCFLAGS="-D warnings" cargo doc -p cuda-coop --no-deps

The tests cover checked access, arbitrary initialized slots, numeric inference,
copies and selection, value-array alignment, and rejected types, extents and
alignments. These host tests and doctests do not establish GPU collective safety.
Rustdoc at ``rust/target/doc/cuda_coop/index.html`` contains the full API reference.
