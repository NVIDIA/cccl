.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Direct Load and Store
=====================

``cuda_coop::load`` fills an existing ``ThreadData`` and returns unit.
``cuda_coop::store`` writes its values and returns unit. Both take the group
first. This initial implementation supports complete, one-dimensional blocks
and the Direct algorithm: each thread owns consecutive elements in blocked
order. Other algorithms, warp groups, and public temporary storage remain
under development.

Use ``use cuda_coop as coop;`` and keep the per-thread item count a const generic
named ``ITEMS_PER_THREAD``. The payload element type is inferred from its source.
The supported types are ``i8``, ``u8``, ``i16``, ``u16``, ``i32``, ``u32``,
``i64``, ``u64``, ``f32`` and ``f64``.

A complete program
------------------

The example is an independent Cargo workspace with a separate generic helper
crate. It copies three block tiles at item counts 1 and 4, including a short last
tile. The consumer has no build script, native shim, provider, or scratch argument.
Its ``cuda-coop`` dependency supplies the native specialization during the normal
Oxide build.

Use the Oxide revision selected by the manifests and ``nightly-2026-10-11``.
The initial native configuration is Linux with CUDA 13.3 and a supported NVIDIA
GPU. From a complete CCCL checkout, with that Oxide toolchain installed:

.. code-block:: shell

   cd rust/cuda-coop/examples/load_store
   CUDA_HOME=/usr/local/cuda-13.3 CUDA_PATH=/usr/local/cuda-13.3 \
     cargo oxide run --arch sm_120 --materialize-cubin

Choose the architecture for your GPU. Materialization produces the native device
artifact during the Cargo build. No separate compilation or linking step is
required. The exact block launch contract is checked when preparing the launch;
``auto_shared`` uses the compiler's inferred requirement, which is zero for
Direct Load/Store.

.. literalinclude:: ../../rust/cuda-coop/examples/load_store/src/main.rs
   :language: rust
   :start-after: // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

The reusable helper preserves the source element type and borrows its payload
read-only for Direct Store:

.. literalinclude:: ../../rust/cuda-coop/examples/load_store/helper/src/lib.rs
   :language: rust
   :start-after: use cuda_coop as coop;

Counts, offsets, and payload validity
-------------------------------------

``LoadOptions::default()`` and ``StoreOptions::default()`` select the full tile
at element offset zero. Named setters select a partial count, a typed Load
default, and an offset. The count describes the whole block's prefix, not a
per-thread count. It must fit the tile and ``i32::MAX``. Offsets are element
indices from the start of the allocation; supply each block's origin explicitly.
Counts, defaults, offsets, and their presence must be uniform across the block.

A masked Load without ``oob_default`` resets every out-of-prefix slot to
unwritten, even if initialized before the Load. ``get`` returns ``None`` for an
unwritten slot; indexing it traps. A default requires an explicit count and has
the source's numeric type. Filled Load initializes every slot, including the
missing tail. Partial Store writes only its selected prefix and requires only
those slots to be initialized.

Direct Store accepts ``&ThreadData``, a shared reborrow of ``&mut ThreadData``,
or a scalar numeric value. A scalar contributes one item per thread. Store
preserves every payload value and validity flag, including initialized slots
outside its write domain. Algorithms that mutate payloads will require mutable
borrows when those algorithms are supported.

Collective ownership and completion
-----------------------------------

Writable kernel parameters use the canonical spelling
``DisjointSlice<T, CollectiveTiles<CAPACITY>>``. The capacity defines each block's
ownership grain as ``block_size * CAPACITY`` elements; it is independent of the
operation's per-thread item count. The carrier keeps the buffer lifetime and
collective ownership without giving each lane an overlapping mutable slice.
The compiler's generated host launch consumes or borrows the actual device
buffers until the operation completes.

The descriptor returned by ``this_block()`` is a query value. Safe collective
calls additionally require a supported Oxide build, an exact launch contract,
uniform control flow, and matching group participation. Compilation diagnoses
lane-varying controls or conflicting resource use. Runtime checks trap invalid
bounds, ownership ranges, counts, or unwritten values selected for Store.
Failures are not transactional across lanes.

To load from a carrier after storing to it, call
``cuda_device::thread::sync_threads()`` and pass ``&mut carrier`` as the Load
source. The compiler requires this completion before conflicting accesses,
including reuse in loops. Ending a Rust borrow or dropping a value is not
completion. The host example calls ``sync()`` before inspecting its output.

Host development
----------------

Host payload tests and rustdoc do not require a GPU or CUDA installation:

.. code-block:: shell

   cd rust
   cargo test --workspace
   cargo clippy --workspace --all-targets -- -D warnings
   RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps

The codegen crate is only a host build dependency. Native headers come from the
same complete CCCL checkout as that dependency; a Python installation or the
consumer's working directory does not select headers. Native builds use
Oxide's selected toolkit. Cargo tracks the provider and declared native inputs;
replacing them with unchanged or older timestamps may require an explicit
rebuild. Crate publication is not supported yet.

The rustdoc entries for ``load``, ``store``, ``LoadOptions``, ``StoreOptions``,
``LoadSource`` and ``StoreValue`` specify their full generic signatures and
supported argument forms. Native correctness and artifact checks are separate
from host compilation. Performance qualification for masked payload access is
ongoing; the current implementation makes no negligible-overhead claim.
