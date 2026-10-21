.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

``cuda.coop`` in Rust
=====================

Write group-first cooperative operations in an ordinary generic Rust kernel.
This kernel copies partial block tiles through a reusable helper in another
crate. The item count stays a const generic from the host launch through the
kernel and helper:

.. literalinclude:: ../../rust/cuda-coop/examples/load_store/src/main.rs
   :language: rust
   :lines: 12-35
   :dedent: 4

The helper uses the following calls for a partial tile. ``ThreadData`` infers its
element type from ``input``. Load fills the payload in place; Store writes the
valid prefix and leaves the rest of the destination unchanged:

.. literalinclude:: ../../rust/cuda-coop/examples/load_store/helper/src/lib.rs
   :language: rust
   :lines: 55-73
   :dedent: 8

The complete :doc:`load_store` example includes the helper, imports, host
allocation and launch. It runs Direct, Striped, Vectorize, Transpose,
WarpTranspose and WarpTransposeTimesliced with one and four items per thread.
These listings come directly from that executable source.

Start with a kernel
-------------------

Use ``use cuda_coop as coop;``. A ``ThreadData::<_, ITEMS_PER_THREAD>`` holds
each thread's values and tracks which slots have been initialized. The group
argument selects the participating threads. ``DisjointSlice`` with
``CollectiveTiles<ITEMS_PER_THREAD>`` supplies collective output ownership
without giving every lane an overlapping mutable Rust slice.

For a block tile, calculate its size as ``block.count() * ITEMS_PER_THREAD``,
its element offset from ``block.rank_at::<coop::Grid>()``, and its valid count
as the remaining input length capped at the tile size. The kernel above shows
that idiom, including a short final tile. Counts and offsets describe the whole
group's tile, and must be uniform across that group.

The current public operations are Load and Store, with the supported groups,
algorithms and numeric types described in :doc:`load_store`. Additional
primitive families are under development. The crate is unpublished and its
interface can change.

Build and run
-------------

Use the Oxide revision selected by the Cargo manifests and the pinned Rust
nightly. The example's initial native configuration is Linux, CUDA 13.3 and an
NVIDIA GPU. From a complete CCCL checkout with the matching Oxide installed:

.. code-block:: shell

   cd rust/cuda-coop/examples/load_store
   CUDA_HOME=/usr/local/cuda-13.3 CUDA_PATH=/usr/local/cuda-13.3 \
     cargo oxide run --arch sm_120 --materialize-cubin

Choose the architecture for your GPU. The kernel's launch contract declares
its exact block shape. Preparing a launch checks that shape and uses the
compiler's inferred shared-memory requirement. The consumer needs no C++ shim,
provider build script, manual linker inputs or scratch-byte calculation.

How compilation works
---------------------

``cargo oxide`` discovers the concrete Load/Store uses, including those inside
generic helpers from another crate. The dependency's host provider specializes
CUB for their types, item counts, algorithms and launch geometry. NVRTC returns
device LTO-IR and the actual C++ temporary-storage layouts. Oxide checks
participation and resource use, plans shared storage and reuse synchronization,
and links the device code into the final CUBIN.

The same build supplies launch geometry and resource facts to the typed host
launcher. Borrow checking remains useful, but collective participation and
scratch completion also require the compiler's checks. See :doc:`provider` for
the request format, native ABI and provider development workflow.

Generic Rust API
----------------

The public API and downstream helpers use Rust types, traits and const
generics. Element types are inferred from typed producers, and ordinary calls
infer the algorithm from their options. Supported element types and algorithms
are deliberately constrained by the API's traits.

Each concrete native specialization gets a generated device C ABI wrapper at
build time. This is an implementation boundary, not a finite set of functions
that users select or declare. The user's source stays generic Rust.

Read next
---------

.. toctree::
   :maxdepth: 1

   payload
   groups
   load_store
   provider

Generate the API reference with ``cargo doc --workspace --no-deps`` from
``rust/``; ``cuda_coop`` contains the public item documentation. The Rust
workspace is separate from CCCL's CMake build, and existing C++ users do not
need a Rust toolchain.
