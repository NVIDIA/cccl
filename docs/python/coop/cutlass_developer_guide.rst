.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _cuda.coop.cutlass.developer_guide:

CUTLASS Developer Guide
=======================

CuTe compiles your kernel and its control flow. For each cooperative
primitive, ``cuda.coop.cutlass`` generates a C++ device function that calls
CUB or CUDAX. NVRTC compiles those functions to LTO-IR, which CuTe links into
the kernel before it runs. The implementation calls these generated
functions *providers*.

The :doc:`shared overview <../coop>` introduces groups, per-thread items, and
the common API. The :doc:`CUTLASS Programming Guide <../coop_cutlass>` covers
writing kernels and choosing CUTLASS-specific controls. Here, an executable
tile-copy example shows how a call becomes a compiled device function and
where to look when changing the implementation. Numba-CUDA-MLIR has its own
:doc:`Developer Guide <developer_overview>` and
:doc:`Programming Guide <programming_guide>`.

Following a tile copy
---------------------

The executable Load/Store example launches one ``(8, 4, 1)`` block. Each of
its 32 threads owns two items, for a tile of 64 integers. ``module`` selects
the common or qualified API; both use this kernel:

.. literalinclude:: ../../../python/cuda_coop/examples/cutlass/block_load_store.py
   :language: python
   :start-after: docs: start cutlass-block-load-store
   :end-before: docs: end cutlass-block-load-store

Load reads 45 items beginning at source offset 3 and fills the other payload
items with ``-7``. Store writes the first 53 items beginning at destination
offset 5. Each thread calls both primitives, including threads whose
items are outside the valid prefix. Load mutates ``payload`` and returns
``None``; Store leaves the payload unchanged.

The :download:`complete example
<../../../python/cuda_coop/examples/cutlass/block_load_store.py>` includes
allocation, launch, cleanup, and an independent NumPy result check. From the
repository root, run it in the compatible environment described in the
programming guide:

.. code-block:: bash

   python python/cuda_coop/examples/cutlass/block_load_store.py

CuTe owns the ``@cute.kernel`` and ``@cute.jit`` functions, their Python
control flow, and the launch. Each primitive is a device call inside the
same kernel.

.. _coop-cutlass-compiler-requirements:

Compiler requirements
---------------------

The CuTe compiler must support external NVIDIA LTO-IR linking, a scoped hook
at the end of tracing, and access to the active compiler environment. It must
also supply exact block, grid, and cluster dimensions and launch flags.
``cutlass/_compiler/_runtime.py`` checks for the required Python interfaces
during registration. Compilation and execution tests check that the
interfaces work together with NVRTC and the final linker.

.. _registration-and-compiler-ownership:

Registration and compiler selection
-----------------------------------

``coop.register("cutlass")`` checks the runtime and registers the CUTLASS
adapter with ``cuda.coop``. Importing ``cuda.coop.cutlass`` has the same
effect. Importing only ``cuda.coop`` leaves the compiler and CUDA bindings
unloaded.

A call through the common API must select the backend for the compiler
tracing the kernel. CUTLASS registers a predicate that compares the active
environment with CuTe's initialized environment. The dispatcher calls it
without creating another compiler or importing another runtime. An explicit
private ``_compiler_scope`` takes precedence, so the Numba compiler can
select its own backend even when CUTLASS is installed. If two predicates
claim the same active environment, dispatch fails.

The selection code is in ``_core/api/_dispatch.py`` and
``cutlass/_compiler/_activation.py``. If a CUTLASS runtime check fails,
registration raises an error while leaving the common API available. Tests
check that a later registration attempt can succeed.

Shared core and primitive families
----------------------------------

The common API is exposed through ``cuda.coop`` and implemented in
``_core/api/``. The private ``_core/`` package also holds the planners and
other implementation shared by both backends.

A :term:`family` groups related primitives and their implementation. The
Scan family includes ``scan``, ``inclusive_scan``, ``exclusive_scan``,
``inclusive_sum``, and ``exclusive_sum``. Its shared declarations live in
``_core/api/scan.py`` and ``scan.pyi``, with planning in
``_core/group/scan.py``. CUTLASS adds its entry points and type declarations
in ``cutlass/_group_scan.py`` and ``_group_scan.pyi``, and adapts the plan to
CuTe in ``cutlass/_lowering/_scan.py``.

.. _from-a-call-to-a-typed-provider-request:

Planning the CUB call
---------------------

For Load, ``cutlass/_group_load_store.py`` first validates the group, payload,
and controls. It classifies a control as omitted, a compile-time constant,
or a runtime value. The lowering then reads the CuTe memory operand's
element type and checks that its layout exposes the contiguous pointer
required by CUB.

The shared planner uses those arguments and the kernel's launch dimensions
to choose a CUB specialization. It also determines which threads must
participate and what scratch storage and barriers the call needs. Numba uses
this planner too. The CUTLASS lowering puts the plan and CuTe scalar type
into a provider request, which the renderer uses to generate the wrapper.

In the example, the default ``direct`` algorithm selects CUB Block Load and
Store for a 32-thread block with two integers per thread. Thread rank
``t`` owns tile items ``2 * t`` and ``2 * t + 1``. The provider source embeds
the static offsets, valid counts, and fill value. A runtime control instead
becomes a typed wrapper argument.

The renderer generates a device wrapper with an ordinary callable symbol.
CuTe's ``cute.ffi`` emits the call to that symbol. Load uses a result buffer
to return its items, then updates the ``ThreadData`` values after the FFI
call is successfully emitted. Store passes the existing items to its
provider. The plan identifies equivalent requests so they can share one
generated function and cached artifact.

Other primitive families use the same approach. For Reduce, the planner
chooses CUDAX for supported full-group reductions. It chooses CUB when the
call supplies a valid-prefix count or a block algorithm. The Reduce lowering
then generates the wrapper and adapts CuTe's values to its arguments.

Exact launch facts
-------------------

``cutlass/_compiler/_launch.py`` reads the compiler-provided exact block,
grid, and cluster dimensions and the cooperative/cluster launch flags. The
adapter records their origin in the shared ``LaunchFacts`` object. If a
dimension or flag is unavailable, it stays unknown in that object. Planning
fails if the primitive needs the missing value.

For the tile copy, block dimensions determine both the CUB specialization
and linear rank: ``x + block_x * (y + block_y * z)``. Warp primitives also
need the exact block size to establish complete physical warps and allocate
one scratch slice per group. Cluster primitives need consistent cluster
dimensions and launch mode.

Do not substitute ``maxntid`` for exact dimensions: an upper bound does not
prove the number of participating threads. The adapter does not infer
launches from Python frames or a user-maintained launch description.

One provider bundle per trace
------------------------------

Each trace has a provider session keyed by its compile options and MLIR
module. The session deduplicates equivalent requests and records deferred
scratch uses. Load and Store in the example contribute to the same bundle;
finalization compiles the two providers together.

CuTe's scoped trace-finalization hook finds the session belonging to the
module being finalized. It leaves an unrelated or nested module's session
alone. The hook renders the requests into one C++ translation unit, compiles
it to NVIDIA LTO-IR, and attaches the artifact to that GPU module's
``link-libraries`` attribute. The CuTe compiler and final linker then resolve
the ``cute.ffi`` calls while compiling the enclosing kernel.

Header and toolkit discovery reuse the neutral helpers under ``_headers``.
The compiler libraries for the selected toolkit are loaded before importing
the NVRTC bindings, and the loaded NVRTC version is checked against that
selection. A mismatch reports an error before provider compilation. The
final linker must also accept the generated LTO-IR; successful NVRTC
compilation alone does not establish that compatibility.

The cache identity includes the rendered source, header identity, compiler
options, architecture, and resolved toolkit/compiler information. Cache
loads validate the artifact and any recorded scratch layouts. Cache writes
use a lock and atomic replacement. The LTO-IR file remains available until
the final linker has read it. Finalization removes stale managed library
paths from persistent compiler options so the next trace cannot link an
earlier bundle by mistake.

Failed emission restores the provider-session snapshot, including its
pending scratch uses. Finalization removes only its own session. Lifecycle
tests exercise failed compilation or linking followed by a successful retry,
as well as repeated and nested compilation. These checks matter because a
single Python process can compile many kernels through the same CuTe DSL.

Scratch allocation after tracing
---------------------------------

Direct, striped, and vectorized Load/Store do not use scratch. Their provider
ABI omits a storage pointer, and passing a ``TempStorage`` descriptor does
not add a reuse barrier or change the storage-free provider's identity.

Algorithms that need scratch record its use while tracing. The finalizer
adds ``sizeof`` and ``alignof`` probes to the same NVRTC program as the
provider bundle. The resulting C++ layouts determine the allocation; the
adapter does not estimate storage sizes from the element count.

Deferred events let one explicit descriptor serve several calls. Shared
storage uses enough capacity for the largest requirement, while exclusive
storage assigns separate slices to distinct call sites. Requested alignment
is a minimum. An explicit capacity that cannot accommodate its uses is an
error. Once the layouts are known, the finalizer materializes shared-memory
allocations and replaces the trace's storage placeholders.

Exclusive slices still need reuse synchronization when one call site runs
again in a loop. Automatic trailing synchronization is the default for both
sharing modes. With ``auto_sync=False``, the kernel must call
``storage.sync()`` before reuse. Physical and logical warp primitives use
independent per-group storage and the appropriate warp mask rather than a
block barrier. See the programming guide for each family's explicit-storage
support and participation rules.

Finding the implementation
---------------------------

Paths below are relative to ``python/cuda_coop/cuda/coop``. Shared primitive
semantics and planner changes belong in ``_core``; CuTe value adaptation and
provider emission belong under ``cutlass``.

.. list-table:: Source map
   :header-rows: 1
   :widths: 36 64

   * - Area
     - Starting points
   * - Root selection and registration
     - ``_registration.py``, ``_core/api/_dispatch.py``,
       ``cutlass/_compiler/_activation.py``, ``cutlass/_compiler/_runtime.py``
   * - User-facing values and groups
     - ``cutlass/_thread_data.py``, ``cutlass/_thread_group.py``,
       ``cutlass/_temp_storage.py``, and their ``.pyi`` files
   * - Family validation and lowering
     - ``cutlass/_group_load_store.py`` and
       ``cutlass/_lowering/_load_store.py``; the Reduce, Scan, Exchange,
       Shuffle, and Merge Sort files follow the same organization
   * - Launch facts and provider sessions
     - ``cutlass/_compiler/_launch.py``, ``cutlass/_compiler/_state.py``,
       ``cutlass/_compiler/_finalize.py``
   * - Rendering, compilation, and artifacts
     - ``cutlass/_compiler/_rendering.py``, ``cutlass/_compiler/_bundle.py``,
       ``cutlass/_compiler/_nvrtc.py``, ``cutlass/_compiler/_cache.py``
   * - Scratch layout and materialization
     - ``cutlass/_compiler/_layout.py``, ``cutlass/_compiler/_storage.py``

Checking a change
-----------------

Start with the affected family under
``python/cuda_coop/tests/backends/cutlass``. Unit tests check plan selection,
argument contracts, and state transitions. Compile tests exercise real CuTe
traces, launch metadata, FFI emission, and lifecycle recovery. Runtime tests
compare results and layouts against independent references, including
partial tiles and repeated scratch reuse.

For a Load/Store or lifecycle change, the following are useful entry points
from the repository root with the package and compatible compiler installed:

.. code-block:: bash

   python -m pytest -q python/cuda_coop/tests/backends/cutlass/unit/test_load_store_plans.py
   python -m pytest -q python/cuda_coop/tests/backends/cutlass/compile/test_compiler_lifecycle.py
   python -m pytest -q python/cuda_coop/tests/backends/cutlass/runtime/test_block_load_store.py
   python -m pytest -q python/cuda_coop/tests/backends/cutlass/runtime/test_block_algorithms.py

The compile and runtime examples above require a compatible CUDA environment
and GPU. Inspect skips when assessing coverage: an absent optional compiler
does not establish a passing compiler integration. Fresh-process tests cover
both import orders and explicit registration; packaging tests under
``tests/packaging`` check installed module origins and qualified typing.
Shared dispatcher or planner changes also require the affected Numba tests.

To inspect the generated C++, set ``CUDA_COOP_SOURCE_DUMP_DIR`` before
compiling a kernel. The dump includes the source on provider-cache hits too.
``CUDA_COOP_CUTLASS_PROVIDER_CACHE_DIR`` selects the provider artifact cache
when a test needs an isolated cache directory.

For generated-code claims, inspect the final linked cubin. The block
algorithm tests include provider-call elimination and scratch/barrier checks
using ``cuobjdump``; provider source or intermediate PTX alone cannot prove
the final result. Run focused Compute Sanitizer race checks for changes to
scratch allocation or synchronization. Keep numerical correctness,
generated-code evidence, and public-package qualification as separate checks.
