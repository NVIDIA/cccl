.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Block and Warp Scan
===================

``scan`` computes a prefix within each participating group and returns a fresh
result. Its defaults select an exclusive sum: the first output is zero, and
each following output sums the preceding inputs. ``exclusive_sum``,
``inclusive_sum``, ``exclusive_scan`` and ``inclusive_scan`` provide the same
operation with their mode or operator fixed by the function name. All five
functions take the group first and preserve the input.

A Block accepts a scalar or a shared or mutable borrow of ``ThreadData``.
The result has the same element type and, for a payload, the same item count
and alignment. Block payloads use blocked order: each thread contributes its
consecutive items, followed by the next thread in x-fastest rank order.
Complete one-, two- and three-dimensional blocks are supported. Each block's
prefix starts independently; this is not a device-wide scan.

Physical Warp and logical lane groups of widths 1, 2, 4, 8, 16 and 32 accept
scalar inputs and return scalars. Prefixes restart at each group's first lane.
The enclosing block must contain complete physical warps. For example:

.. code-block:: rust

   let group = coop::this_warp().group_by::<8>();
   let prefix = coop::exclusive_sum(group, value, Default::default());

A complete program
------------------

This independent Cargo workspace uses a generic helper in another crate. The
kernel and helper share the const-generic ``ITEMS_PER_THREAD``; the payload
element type and fresh result type are inferred. The program checks one and
four items per thread across three Block tiles, including a short final tile.

.. literalinclude:: ../../rust/cuda-coop/examples/scan/src/main.rs
   :language: rust
   :start-after: // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

The helper initializes the missing tail with zero before Scan, then stores only
the valid prefix. Load returns unit; Scan returns a new payload; Store reads
that result through a shared borrow.

.. literalinclude:: ../../rust/cuda-coop/examples/scan/helper/src/lib.rs
   :language: rust
   :start-after: // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Use the Oxide revision selected by the manifests and ``1.99.1``.
With that toolchain installed, run from a complete CCCL checkout:

.. code-block:: shell

   cd rust/cuda-coop/examples/scan
   CUDA_HOME=/usr/local/cuda-13.3 CUDA_PATH=/usr/local/cuda-13.3 \
     cargo oxide run --arch sm_120 --materialize-cubin

Choose the architecture for your GPU. The dependency supplies its host provider;
the consumer needs no build script, C++ shim, manual linker inputs or scratch
size. ``auto_shared`` uses the compiler's complete shared-memory plan.

Modes, operators, and seeds
---------------------------

``ScanOptions::default()`` selects exclusive mode, ``Sum``, the default Block
algorithm, an implicit zero seed and compiler-managed storage. Consuming
setters select compile-time options and the runtime seed:

.. code-block:: rust

   let inclusive = coop::scan(block, &values,
       coop::ScanOptions::default().mode(coop::Inclusive));
   let products = coop::exclusive_scan(block, &values,
       coop::ScanOptions::default()
           .scan_op(coop::Multiplies).initial_value(1));
   let maxima = coop::inclusive_scan(block, &values,
       coop::ScanOptions::default()
           .scan_op(coop::Max).algorithm(coop::RakingMemoize));

``Exclusive`` emits the prefix before the current input; ``Inclusive`` includes
the current input. An exclusive seed participates in every prefix and has
exactly the input element type. Exclusive ``Sum`` may omit its seed; every
other exclusive operator requires one. Inclusive operations reject a seed.
The sum functions fix ``Sum`` and do not accept a replacement operator or seed;
the named exclusive and inclusive functions do not accept a mode override.

The supported elements are ``i8``, ``u8``, ``i16``, ``u16``, ``i32``, ``u32``,
``i64``, ``u64``, ``f32`` and ``f64``. Integers support ``Sum``, ``Multiplies``,
``Min``, ``Max``, ``BitAnd``, ``BitOr`` and ``BitXor``. Floating types support the
first four. These operators use the input dtype; the API does not widen narrow
integers. CUB may regroup associative operations, so floating-point results
need not match a sequential host sum exactly. Arbitrary callables are not yet
supported.

Validity and participation
--------------------------

Every input slot must be initialized. Scan does not accept a valid count or
skip unwritten items. A masked Load without a default can leave such slots;
Scan traps before reading an unwritten input. For a partial tile, use a filled
Load or initialize each missing slot explicitly. Choose a fill appropriate for
the operator and the output you need. Every returned slot is initialized,
including outputs corresponding to the filled tail.

All group members must reach the same collective. Seeds and storage selections
must be uniform within the group, though separate logical groups may use
different seeds. Input values remain lane-local. Rust borrowing preserves local
ownership; the supported compiler also checks collective participation and
shared-storage use.

Algorithms and temporary storage
--------------------------------

Block Scan supports ``Raking`` (the default), ``RakingMemoize`` and
``WarpScans``. Select one with ``.algorithm(...)``. ``WarpScans`` requires a
total block thread count divisible by 32. Warp Scan uses its implicit native
algorithm and storage; it rejects an explicit Block algorithm or
``TempStorage`` argument.

Block calls can use the same public temporary-storage descriptor as
:doc:`Load/Store <load_store>`:

.. code-block:: rust

   let mut scratch = coop::TempStorage::new().auto_sync();
   let prefixes = coop::exclusive_scan(block, &values,
       coop::ScanOptions::default()
           .initial_value(seed).temp_storage(&mut scratch));

Omitting a descriptor requests managed synchronization before scratch reuse.
An explicit descriptor defaults to manual completion; ``auto_sync()`` adds a
trailing barrier after a scratch-using operation. The provider queries the
actual CUB layout, and Warp storage is replicated for each group. Scan itself
does not promise completion of unrelated global Load/Store accesses. Retain
the required group barriers when reusing a destination or explicit scratch.
