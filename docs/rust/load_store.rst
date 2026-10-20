.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Block Load and Store
====================

``cuda_coop::load`` fills an existing ``ThreadData`` and returns unit.
``cuda_coop::store`` writes its values and returns unit. Both take the group
first. Complete one-, two- and three-dimensional blocks support Direct and
Transpose algorithms. Both Load algorithms produce blocked
payloads, where each thread owns consecutive elements. Other algorithms and
warp groups remain under development.

Use ``use cuda_coop as coop;`` and keep the per-thread item count a const generic
named ``ITEMS_PER_THREAD``. The payload element type is inferred from its source.
The supported types are ``i8``, ``u8``, ``i16``, ``u16``, ``i32``, ``u32``,
``i64``, ``u64``, ``f32`` and ``f64``.

A complete program
------------------

The example is an independent Cargo workspace with a separate generic helper
crate. It copies three block tiles with Direct and Transpose at item counts 1
and 4, including a short last tile. It also uses explicit automatic scratch in
a repeated helper call. The consumer has no build script, native shim, provider,
or scratch-size argument.
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
Direct Load/Store and includes the planned shared storage for Transpose.

.. literalinclude:: ../../rust/cuda-coop/examples/load_store/src/main.rs
   :language: rust
   :start-after: // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

The reusable helper preserves the source element type. Its algorithm-generic
form borrows the payload mutably because Transpose Store permutes it.
Direct-only code may pass a shared payload borrow:

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

For block shape ``[X,Y,Z]``, thread rank is x-fastest:
``t = x + X * (y + Y * z)``. A blocked payload slot ``i`` addresses
``offset + t * ITEMS_PER_THREAD + i``. The exact shape is part of the kernel
launch contract; another shape with the same thread count is rejected.
Use ``LaunchConfig2D`` or ``LaunchConfig3D`` for the corresponding launch domain.
Declare the matching kernel contract, for example
``#[launch_contract(domain = 3, block = (8, 4, 4), dynamic_shared = inferred)]``.
The block origin in a multidimensional grid uses x-fastest block rank as well;
for a collective carrier, include that rank times its ownership grain
``block_size * CAPACITY`` in the element offset (see `Collective ownership and
completion`_). Read-only slice loads have no ownership grain.

A masked Load without ``oob_default`` resets every out-of-prefix slot to
unwritten, even if initialized before the Load. ``get`` returns ``None`` for an
unwritten slot; indexing it traps. A default requires an explicit count and has
the source's numeric type. Filled Load initializes every slot, including the
missing tail. Partial Store writes only its selected prefix and requires only
those slots to be initialized.

Direct Store accepts ``&ThreadData``, a shared reborrow of ``&mut ThreadData``,
or a scalar numeric value. A scalar contributes one item per thread. Direct Store preserves every payload value and validity flag, including
initialized slots outside its write domain.

Selecting Transpose
-------------------

Select the algorithm through the trailing options. Selection is a Rust type,
so each call compiles one algorithm; ordinary defaults continue to select Direct.

.. code-block:: rust

   coop::load(block, input, &mut values,
       coop::LoadOptions::default().algorithm(coop::Transpose));
   coop::store(block, output, &mut values,
       coop::StoreOptions::default().valid_items(count).algorithm(coop::Transpose));

Transpose uses CUB's coalesced accesses and shared block exchange. Its Store
requires a mutable payload borrow and permutes every value and initialized bit
from blocked into striped order. For a block of ``B`` threads and ``I`` items
per thread, destination lane/item ``(t, i)`` receives the old blocked slot at
linear index ``B*i+t``. This mutation occurs even for a zero-count Store; a
partial count only controls the global writes. Initialized slots outside the
written prefix remain initialized at their permuted positions. Scalar Store
uses a one-item temporary and accepts either algorithm.

The compiler allocates the provider's complete shared storage, including the
validity plane when needed, and inserts completion before reusing managed
scratch. This does not add an unconditional trailing barrier. ``auto_shared``
uses the inferred total for the kernel.

Explicit temporary storage
--------------------------

Construct ``TempStorage`` inside the kernel and pass a mutable borrow through
the options. Default storage shares one slice between its uses and requires
an explicit full-block barrier before reuse. Omitting ``temp_storage`` requests
compiler-managed reuse synchronization instead.

.. code-block:: rust

   let mut scratch = coop::TempStorage::new().auto_sync();
   coop::load(block, input, &mut values,
       coop::LoadOptions::default().algorithm(coop::Transpose)
           .temp_storage(&mut scratch));
   coop::store(block, output, &mut values,
       coop::StoreOptions::default().algorithm(coop::Transpose)
           .temp_storage(&mut scratch));

The consuming ``auto_sync()`` builder inserts a trailing block barrier after
each operation that actually uses scratch. Direct Load/Store receives no barrier
or allocation from this policy. Options carrying a mutable storage borrow are
consumed by the operation; construct fresh options with an ordinary reborrow
for each call, including inside loops.

``exclusive()`` gives separate static primitive and reservation sites separate
slices. Repeated execution of one site still reuses its slice. Each static
constructor site denotes one descriptor, including in a loop; different
constructor sites receive independent storage. Independent scratch does not
complete conflicting accesses to the same global-memory carrier.

``with_capacity::<BYTES>()`` specifies a positive capacity for the complete
descriptor plan, including exclusive slices and padding. An undersized request
is a compilation error. ``with_alignment::<ALIGN>()`` selects a positive
power-of-two minimum alignment for the allocation base; each exclusive slice
also satisfies its own operation or reservation requirement. Omitted capacity
and alignment are inferred. Consuming builders retain the original descriptor
identity and must not change its policy after use. No destructor synchronizes.

Typed reservations
------------------

Descriptors without auto-sync support ``reserve::<T, N>()`` and
``reserve_aligned::<T, N, ALIGN>()``. ``N`` is a positive element count, and the
aligned form requests a positive power-of-two minimum for that view. Only the
ten numeric types listed above are supported. Select the element type in the reservation, for example
``scratch.reserve::<i32, 128>()``.

Reservations return non-Copy tokens with ``as_mut_ptr()`` and consuming
``release()`` methods. They expose no initialized Rust value, reference, slice,
or safe indexing operation. The token does not hold a descriptor-long mutable
borrow, allowing simultaneous exclusive reservations.

All element access is unsafe. The caller must establish bounds, initialization
before reads, aliasing, non-conflicting accesses, and phase synchronization.
Saved pointers must not be accessed after release, during overlapping reuse,
or after kernel exit. Complete asynchronous accesses before ending their view.
The compiler checks planned origins and live overlap; it does not prove arbitrary
unsafe pointer code.

To reuse a shared region, release its reservation, then synchronize the full
block before the next overlapping operation. Release does not emit a barrier or
complete global-memory accesses. A barrier before release is insufficient, and
dropping or forgetting the token does not release it. Typed reservations cannot
be combined with automatic synchronization on their descriptor.

Collective ownership and completion
-----------------------------------

Writable kernel parameters use the canonical spelling
``DisjointSlice<T, CollectiveTiles<CAPACITY>>``. The capacity defines each block's
ownership grain as ``block_size * CAPACITY`` elements; it is independent of the
operation's per-thread item count. The carrier keeps the buffer lifetime and
collective ownership without giving each lane an overlapping mutable slice.
The compiler's generated host launch consumes or borrows the actual device
buffers until the operation completes.

For a grid of ``G`` blocks with ownership grain ``C``, block rank ``r`` owns
``[r*C, (r+1)*C)`` in each period of ``G*C`` elements. An offset may start inside
that region, and its nonempty selected range must stay inside the same region.
This also supports grid-stride traversal through later periods.

The descriptor returned by ``this_block()`` is a query value. Safe collective
calls additionally require a supported Oxide build, an exact launch contract,
uniform control flow, and matching group participation. Compilation diagnoses
lane-varying controls or conflicting resource use. Runtime checks trap invalid
bounds, ownership ranges, counts, or unwritten values selected for Store.
Failures are not transactional across lanes.

Use ``cuda_device::thread::sync_threads()`` between conflicting accesses to a
carrier: Store followed by Load or Store, or Load followed by Store. Pass
``&mut carrier`` as the Load source. The compiler requires completion, including reuse in loops. A genuine planned
scratch-reuse barrier can complete earlier accesses; code should not rely on an
algorithm merely being named Transpose to provide a trailing barrier. Ending a Rust borrow or dropping a value is not
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
``LoadSource``, ``StoreValue``, ``Algorithm``, ``Direct`` and ``Transpose`` specify their full generic signatures and
supported argument forms. Native correctness and artifact checks are separate
from host compilation. Performance qualification for masked payload access is
ongoing; the current implementation makes no negligible-overhead claim.
