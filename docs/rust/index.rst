.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Rust development
================

The Rust workspace contains per-thread payloads, CUDA group queries, and the
native host compiler foundation for ``cuda.coop``. The ``cuda-coop`` crate supplies checked
``ThreadData`` values, physical/mapped hierarchy queries, and Direct/Transpose block
Load/Store; see :doc:`payload`, :doc:`groups`, and :doc:`load_store`. Its private ``cuda-coop-codegen`` crate
specializes CUB block operations and returns
linkable device LTO-IR with exact temporary-storage requirements. It supports
compiler integration development. Additional collective algorithms and public
scratch storage remain under development. The crate is unpublished and its interface can change.

Device code uses the pinned Oxide dependency and ``1.99.1``.
The host provider supports Rust 1.85 independently; validate it with
``RUSTC_BOOTSTRAP=-1 cargo +1.85.0 test -p cuda-coop-codegen --locked``
from ``rust/``; this disables unstable features for the provider check.

The workspace is separate from CCCL's CMake build. Building the existing C++
libraries does not require Rust, and the provider does not depend on Python.

Supported provider forms
------------------------

Each request describes a complete, one-dimensional block and a positive
compile-time item count per thread. The original raw ``i32`` recipes support
these forms:

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Operation
     - Behavior
     - Payload requirement
   * - Full Load
     - Fills every payload slot in blocked order
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

Both Direct and Transpose algorithms are supported. Direct accesses no shared
scratch. Transpose Store redistributes blocked payloads across lanes into striped
order, retaining initialized values in every slot, and uses the exact CUB storage
layout. Compiler integrations must enforce collective participation, memory
ownership, valid bounds, and synchronization before scratch reuse.

The separate validity-aware Direct and Transpose recipes support all ten numeric payload
types: ``i8``, ``u8``, ``i16``, ``u16``, ``i32``, ``u32``, ``i64``, ``u64``,
``f32`` and ``f64``. ``Request::new_typed`` selects the exact type. These recipes
take opaque value storage and a packed validity bitmap. Full and
default-filled Load initialize every slot. Masked Load without a default copies
only the loaded prefix and resets all other flags to zero. Full and partial
Store require initialized slots in their write domain. Direct preserves all
payload bytes and validity bits; Transpose permutes both into striped order,
including slots outside its write prefix. A failing Store traps; it is not transactional
across lanes. Partial Store forms destination pointers only within its valid
prefix, including when the allocation is shorter than a complete tile.

The public Direct and Transpose API uses
the same ``ThreadData`` representation without exposing an initialized slice or
a public native layout.

Compile a provider
------------------

Run Cargo from within ``rust/`` using the pinned toolchain in
``rust/rust-toolchain.toml``. The workspace
configuration enables the unstable Rust features required by Oxide and
``cuda-coop`` on that compiler. Cargo finds this configuration from the working
directory; ``--manifest-path`` alone does not select it. Host tests, lint checks, and rustdoc builds
need no CUDA installation. Running native compilation requires
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
is ``compute_120``. These checks compile the supported Direct/Transpose forms at
item counts 1 and 4, compare repeated LTO output, and check exact layout decoding
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
