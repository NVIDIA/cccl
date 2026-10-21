.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Rust development
================

The Rust workspace contains per-thread payloads, CUDA group queries, and the
native host compiler foundation for ``cuda.coop``. The ``cuda-coop`` crate supplies checked
``ThreadData`` values, physical/mapped hierarchy queries, block Load/Store with
Direct, Striped, Vectorize, Transpose, WarpTranspose and WarpTransposeTimesliced,
explicit block storage, and physical-Warp Direct; see :doc:`payload`,
:doc:`groups`, and :doc:`load_store`. Its private ``cuda-coop-codegen`` crate
specializes CUB block operations and returns
linkable device LTO-IR with exact temporary-storage requirements. It supports
compiler integration development. Additional collective algorithms and Warp
capabilities remain under development. The crate is unpublished and its interface
can change.

Device code uses the pinned Oxide dependency and ``nightly-2026-10-11``.
The host provider supports Rust 1.85 independently; validate it with
``cargo +1.85.0 test -p cuda-coop-codegen --locked`` from ``rust/``.

The workspace is separate from CCCL's CMake build. Building the existing C++
libraries does not require Rust, and the provider does not depend on Python.

Supported provider forms
------------------------

Each request carries exact ``[X,Y,Z]`` block dimensions and a
positive compile-time item count per thread. ``Request::new`` and
``Request::new_typed`` retain their one-dimensional ``[B,1,1]`` shape;
``new_with_shape`` and ``new_typed_with_shape`` accept all three dimensions.
Thread rank is x-fastest. Validity-aware Direct requests also support physical
warps when the block contains complete warps. Each warp receives a separate
32-lane tile, with no shared scratch. The original raw ``i32`` recipes support
these forms:

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Operation
     - Behavior
     - Payload requirement
   * - Full Load
     - Fills every payload slot in the selected arrangement
     - Writable storage
   * - Default-filled Load
     - Loads a valid prefix and fills the suffix with a supplied value
     - Writable storage
   * - Full Store
     - Writes every payload slot to memory
     - Every slot initialized
   * - Partial Store
     - Writes a valid prefix; leaves the destination suffix unchanged
     - Every slot initialized

Direct, Striped and Vectorize access no shared scratch. Direct and Vectorize
use blocked payloads; Striped uses striped payloads. Full Vectorize operations
use wide accesses when the type, item count and adjusted pointer permit them;
misaligned pointers and counted overloads use Direct accesses. Transpose Store
redistributes blocked payloads across lanes into striped order and uses the
queried CUB storage layout. WarpTranspose and WarpTransposeTimesliced exchange
within physical warps and require a complete block whose thread count is
divisible by 32. The timesliced form reuses one warp's CUB exchange buffer.
Compiler integrations enforce collective
participation, memory ownership, valid bounds and completion before scratch reuse.

The validity-aware recipes support all six algorithms and all ten numeric
payload types: ``i8``, ``u8``, ``i16``, ``u16``, ``i32``, ``u32``, ``i64``,
``u64``, ``f32`` and ``f64``. ``Request::new_typed`` selects the exact type.
Separate packed validity bits track initialized slots. Full and default-filled
Load initialize every slot; masked Load resets slots outside its selected
prefix to unwritten. Store checks initialization throughout its write domain.
Direct, Striped and Vectorize preserve payloads and bits; the three transposing
algorithms permute both, including slots outside the prefix. Their complete
scratch layouts include a validity plane when needed. For counted warp-transpose
Store with more than one item per thread, that plane covers every warp even
when the CUB exchange buffer is timesliced. A failing Store traps and is not transactional
across lanes. Counted Store forms destination pointers only within its valid
prefix, including when the allocation is shorter than a complete tile.

The public API uses this representation without exposing an initialized slice
or a public native layout.

Compile a provider
------------------

Use the pinned toolchain in ``rust/rust-toolchain.toml``. Host tests, lint checks,
and rustdoc builds need no CUDA installation. Running native compilation requires
a Linux CUDA toolkit with NVRTC LTO support; CUDA 13.3 is the initial validation
toolkit. A GPU and CUDA context are not required.

From the CCCL checkout:

.. code-block:: shell

   cd rust
   cargo test --workspace
   cargo clippy --workspace --all-targets -- -D warnings
   RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
   cargo run -p cuda-coop-codegen --example specialize -- \
     /usr/local/cuda-13.3 compute_120 /tmp/block_load.ltoir

The example below is the executed Cargo source. It obtains its CCCL source root
from its build location, passes an explicit toolkit root, compiles a filled Load,
and writes the resulting LTO-IR. It does not link or launch a device kernel.

.. literalinclude:: ../../rust/cuda-coop-codegen/examples/specialize.rs
   :language: rust
   :start-after: //! Compile a host-side provider without a CUDA context or a device consumer.

Native regression checks are opt-in:

.. code-block:: shell

   CUDA_HOME=/usr/local/cuda-13.3 cargo test --workspace -- --include-ignored

``CUDA_COOP_TEST_TARGET`` can select another supported NVRTC target; its default
is ``compute_120``. These checks compile the supported algorithm forms at
item counts 1 and 4 plus odd-count Vectorize cases, compare repeated LTO output,
and check exact layout decoding
and specialization identity. They do not execute GPU kernels.

Compiler integration and API reference
--------------------------------------

``Provider::from_checkout`` snapshots headers from the explicitly selected CCCL
checkout and toolkit, then loads NVRTC from that toolkit's ``lib64`` directory.
Those native libraries are trusted executable compiler inputs. No Python wheel,
working-directory header lookup, or runtime query kernel is involved. Reuse the
provider and batch requests for a common target into one NVRTC compilation.

The same NVRTC program produces both LTO-IR and each CUB ``TempStorage`` layout.
The result distinguishes scratch-free specializations from an empty C++ type's
nonzero ``sizeof``. Input identities include the request, header contents,
provider source, loaded NVRTC/builtins bytes, version, target, and compiler options.
Symbols include a specialization digest. Request planning must assign each
specialization to exactly one batch per final link: a batch is one indivisible
LTO-IR buffer, so partially overlapping batches would duplicate strong definitions.

Rustdoc is the API reference, including the generated ABI's pointer, validity,
mutation, and collective requirements. Open ``rust/target/doc/cuda_coop_codegen/index.html``
after the documentation command above, or run ``cargo doc --no-deps --open`` from
``rust/``. Device discovery, lowering, linking, launch enforcement, and Cargo
invalidation of external compiler inputs belong to the compiler integration.

The Oxide adapter's ``oxide::serve_inputs`` answers
``--oxide-library-inputs-v1`` with the selected header roots and NVRTC library
paths, without loading CUDA. A provider build script dispatches this query
alongside ``--oxide-library-request-v1``. Device registration getters use
``cuda_device::library::registration!("REGISTRATION_ENV_NAME")`` so the frontend
tracks those files, directory contents and selected toolkit before emitting
Cargo dependency information. Host checks and rustdoc do not query CUDA inputs.
Cargo's timestamp freshness rules apply; replacements with unchanged or older
timestamps may require an explicit rebuild.

.. toctree::
   :maxdepth: 1

   payload
   groups
   load_store
