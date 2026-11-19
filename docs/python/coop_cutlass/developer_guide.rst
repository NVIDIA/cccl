.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _cuda.coop.cutlass.developer_guide:

Developer Guide
=======================

.. raw:: html

   <style>
   .coop-cutlass-compiler-tabs { container-type: inline-size; }
   .coop-cutlass-compiler-tabs .coop-cutlass-diagram-mobile {
     display: none;
     max-width: 360px;
     margin-inline: auto;
   }
   #developer-guide code.literal .pre {
     white-space: normal;
     overflow-wrap: anywhere;
   }
   @container (max-width: 760px) {
     .coop-cutlass-compiler-tabs .coop-cutlass-diagram-desktop {
       display: none;
     }
     .coop-cutlass-compiler-tabs .coop-cutlass-diagram-mobile {
       display: block;
     }
   }
   @media print {
     .coop-cutlass-compiler-tabs .coop-cutlass-diagram-desktop {
       display: block;
     }
     .coop-cutlass-compiler-tabs .coop-cutlass-diagram-mobile {
       display: none;
     }
   }
   </style>

CuTe compiles your kernel and its control flow. For each cooperative
primitive, ``cuda.coop.cutlass`` generates a C++ device function that calls
CUB. NVRTC compiles those functions to LTO-IR, which CuTe links into
the kernel before it runs. The implementation calls these generated
functions *providers*.

During tracing, CuTe runs the kernel's Python code with symbolic values
and builds its intermediate representation (IR). A cooperative call at
this stage adds device instructions to that representation. The GPU runs
them later. *Lowering* adapts the call and its values to the generated
provider's argument and result types. A *provider request* records which
C++ function is needed, and a *session* collects those requests for one
trace.

The :doc:`overview <index>` shows a first kernel. :doc:`concepts` introduces
groups and per-thread items. The :doc:`Programming Guide <programming_guide>`
covers writing kernels. The :doc:`API reference <api>` opens with the public
import and documents each operation's controls. Here, the executable Scan
example shows how a call becomes a compiled device function and where to
look when changing the implementation.

For a hands-on tour, follow the :ref:`cuda.coop.cutlass.debugger_walkthrough`.
It uses a 128-thread tile copy to show import, planning, code generation,
linking, and repeated execution.

If CUTLASS and its CuTe DSL are new to you, read :doc:`cutlass_primer`
first. It explains how the DSL traces a kernel, how that differs from
Dynamo, Numba, and Triton, and which DSL facilities this package relies on.
For the CuTe integration mechanisms without the package's
abstractions, see :doc:`cutlass_integration_example`. It calls
``cub::BlockReduce`` from a CuTe kernel in one file and lists the questions
open for CUTLASS reviewers.

.. toctree::
   :maxdepth: 1

   cutlass_primer
   cutlass_integration_example

Following a prefix sum
-----------------------

The executable Scan example launches one ``(8, 4, 2)`` block. By default,
each of its 64 threads owns two items, for a tile of 128 integers. The host
entry point checks the following kernel against NumPy:

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/scan.py
   :language: python
   :start-after: docs: start cutlass-scan
   :end-before: docs: end cutlass-scan

Load fills ``payload`` with consecutive input values. Scan returns a new
payload whose item at position ``i`` is seven plus the sum of all inputs
before ``i`` in blocked order. Input ``[0, 1, 2, 3, ...]`` therefore produces
``[7, 7, 8, 10, ...]``. Store writes those prefixes to the destination.
Every thread calls all three primitives. They reuse one ``TempStorage``
descriptor, with ``auto_sync=True`` synchronizing scratch reuse.

The `complete example on GitHub
<https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/examples/scan.py>`_ includes
allocation, launch, cleanup, and an independent NumPy result check. From the
repository root, run it in an environment that meets the
:ref:`requirements <coop-cutlass-requirements>`:

.. code-block:: bash

   python python/cuda_coop_cutlass/examples/scan.py

CuTe owns the ``@cute.kernel`` and ``@cute.jit`` functions, their Python
control flow, and the launch. Each primitive is a device call inside the
same kernel.

.. _coop-cutlass-compiler-requirements:

Compiler requirements
---------------------

The CuTe compiler must support external NVIDIA LTO-IR linking and
trace-finalization hooks that run after tracing and before module hashing.
It must provide access to the active compiler environment, exact block, grid,
and cluster dimensions, launch flags, and the ``GPUArch`` compile option.
``_compiler/_runtime.py`` checks for the required Python interfaces
when the package is imported. ``compile_operator`` also needs
``cute.compile_to`` and the ``cutlass.compiler`` names ``CuteCompiler``,
``ArtifactType``, ``TargetKind``, ``Var`` and ``DType``, and checks for them
when it is called. Compilation and execution tests check that the
interfaces work together with NVRTC and the final linker.

.. _cuda.coop.cutlass.compiler_flow:

Compiler phases and handoffs
----------------------------

Read each figure from top to bottom. These steps run on the host. CuTe
traces the Python kernel with symbolic values. It compiles and links the
result before the GPU executes it.

The phase map shows where cooperative calls enter this process. Select
**Python classes** to see the classes, data records, and module functions
that implement each step. Function labels include parentheses. A data
record carries information between steps.

.. tab-set::
   :class: coop-cutlass-compiler-tabs

   .. tab-item:: Overview

      .. figure:: visualizations/cutlass-compiler-phases.svg
         :alt: CUTLASS compilation phases, from tracing cooperative calls
               through provider collection, bundle compilation, scratch
               materialization, convergent provider calls, and final linking.
         :width: 100%
         :figclass: coop-cutlass-diagram-desktop

         Tracing records calls. Finalization supplies their definitions and
         completes deferred scratch allocations.

      .. figure:: visualizations/cutlass-compiler-phases-mobile.svg
         :alt: CUTLASS compilation phases arranged vertically, from tracing
               through provider compilation, scratch materialization,
               convergent provider calls, linking, and execution.
         :width: 100%
         :figclass: coop-cutlass-diagram-mobile

         Tracing records calls. Finalization supplies their definitions and
         completes deferred scratch allocations.

   .. tab-item:: Python classes

      .. figure:: visualizations/cutlass-compiler-phases-classes.svg
         :alt: CUTLASS phases with concrete implementation names.
               BundleSession collects provider requests. The trace finalizer
               obtains a BundleCompilation, materializes
               DeferredTempStoragePlan records, and marks provider calls
               convergent before CuTe links the kernel.
         :width: 100%
         :figclass: coop-cutlass-diagram-desktop

         Classes and records describe ownership and data. Module functions
         render, compile, and finalize the provider bundle.

      .. figure:: visualizations/cutlass-compiler-phases-classes-mobile.svg
         :alt: Detailed CUTLASS phases in a vertical layout, including
               BundleSession, provider request records, bundle compilation,
               deferred scratch plans, and convergent provider calls.
         :width: 100%
         :figclass: coop-cutlass-diagram-mobile

         Classes and records describe ownership and data. Module functions
         render, compile, and finalize the provider bundle.

Each cooperative call is planned as CuTe traces it. The lowering already
has the operand types and exact launch facts needed to select its provider.
It adds a request to ``BundleSession``, then emits a symbolic call. CuTe's
trace-finalization hook runs after these requests have been collected and
before device linking.

The next figure expands this boundary. It shows a nonempty session on a
provider-cache miss, with deferred scratch. NVRTC compiles the provider
bundle. CuTe compiles the enclosing kernel and arranges the final link.
The figures show the device-LTO path that produces a cubin. CuTe supplies
kernel LTO-IR and provider LTO-IR to nvJitLink.

.. tab-set::
   :class: coop-cutlass-compiler-tabs

   .. tab-item:: Overview

      .. figure:: visualizations/cutlass-compiler-swimlane.svg
         :alt: Actor swimlanes show CuTe tracing, cuda.coop.cutlass
               collecting providers, NVRTC returning LTO-IR and scratch
               layouts, and CuTe arranging the final device link before
               launch.
         :width: 100%
         :figclass: coop-cutlass-diagram-desktop

         The bundle and layout queries share one NVRTC program. The final
         device link is a later step.

      .. figure:: visualizations/cutlass-compiler-swimlane-mobile.svg
         :alt: CUTLASS compilation handoffs in time order, with CuTe,
               cuda.coop.cutlass, NVRTC, and the device linker identified at
               each step.
         :width: 100%
         :figclass: coop-cutlass-diagram-mobile

         The bundle and layout queries share one NVRTC program. The final
         device link is a later step.

   .. tab-item:: Python classes

      .. figure:: visualizations/cutlass-compiler-swimlane-classes.svg
         :alt: Implementation-level handoffs through BundleSession, the
               trace-finalization hook, BundleCompilation, deferred storage
               planning, convergent provider calls, and GPU module
               link-libraries.
         :width: 100%
         :figclass: coop-cutlass-diagram-desktop

         The finalizer completes scratch operands, marks provider calls
         convergent, and supplies an artifact path. The launch is a later
         step.

      .. figure:: visualizations/cutlass-compiler-swimlane-classes-mobile.svg
         :alt: Detailed CUTLASS handoffs in a vertical layout, from provider
               requests and symbolic calls through NVRTC results, scratch
               operand replacement, convergent provider calls, and final
               linking.
         :width: 100%
         :figclass: coop-cutlass-diagram-mobile

         The finalizer completes scratch operands, marks provider calls
         convergent, and supplies an artifact path. The launch is a later
         step.

Equivalent provider requests share one generated wrapper. Scratch events
retain each traced use, because separate uses can need separate slices.
A valid cache entry supplies both the LTO-IR and the queried layouts, so
that path skips NVRTC compilation. A session with no providers needs no
bundle. Separate trace modules keep separate sessions. A trace module can
contain several kernels. Deferred scratch is allocated separately for each
kernel.

Calls before compiled provider code exists
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A provider request determines a symbol name and an argument/result contract
before NVRTC runs. The renderer uses that name for the C++ definition.
The lowering uses the same name and contract to emit ``cute.ffi`` calls.
This step needs a symbol name. The provider's device code is linked and
loaded later.

At finalization, ``_trace_finalize_hook()`` marks each provider call
convergent, then attaches the compiled bundle to the GPU module's
``link-libraries`` attribute. To mark the calls,
``lower_convergent_provider_calls()`` replaces the ``func.func`` declaration
and each ``func.call`` that ``cute.ffi`` emitted with ``llvm.func`` and
``llvm.call`` operations that carry the ``convergent`` attribute, under the
same symbol. A provider can contain barriers, shuffles, and reductions that
the threads of a group run together. The attribute keeps the optimizer from
copying the call into branches that only some threads take. The hook also
attaches the LTO-IR file of each compiled device operator that the trace
uses.

By default, a direct call to a ``@cute.jit`` function uses CuTe's compile
cache. CuTe then computes its module hash after the hook returns, builds
the module, and checks the cache. ``cute.compile`` turns the cache off and
skips the hash. The device linker resolves each provider call against its
definition and can optimize it with the kernel. The provider can be
inlined, so it need not remain a separate function in the final cubin.
Loading that cubin makes the kernel available to launch.

Scratch addresses follow a separate path. During tracing, calls that need
deferred scratch use placeholder address and size operands. NVRTC supplies
the exact C++ size and alignment. The finalizer inserts shared-memory
allocations and replaces those operands before CuTe compiles the kernel.
Each provider call keeps the symbol name it was emitted with.

What to inspect at each boundary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Trace data and finalization data
   :header-rows: 1
   :widths: 33 67

   * - Object or record
     - What it tells you
   * - ``GroupLoweringPlan``
     - Selected implementation, participation, result, storage, and
       synchronization requirements for one cooperative call.
   * - Provider request, such as ``_CubScanRequest``
     - Plan plus the CuTe types and ABI choices needed to render a
       wrapper and emit its matching call.
   * - ``BundleSession``
     - Unique provider requests, ordered scratch events, and ``TempStorage``
       lifetime events for one trace module. A snapshot restores this
       bookkeeping after failed emission. IR already emitted stays in the
       trace.
   * - ``BundleCompilation``
     - Path to provider LTO-IR and exact C++ layouts indexed by requirement.
       A cache hit and a fresh compilation return the same kind of result.
   * - ``DeferredTempStoragePlan``
     - Allocation size, alignment, and per-use bindings. Materialization
       inserts allocations and replaces each use's address and size.

.. _coop-compiler-ownership:

Compiler integration
--------------------

Importing ``cuda.coop.cutlass`` checks the CuTe runtime through
``_compiler/_runtime.py``. A cooperative call reads the active CuTe compiler
environment and emits into the enclosing kernel trace. Kernel compilation
starts after the import, when CuTe traces the kernel.

The private ``_compiler`` modules own trace sessions, provider bundles,
launch facts, finalization, cache artifacts, and deferred storage. The
``_planning`` modules select CUB specializations and their participation,
result, storage, and synchronization contracts. The ``_headers`` package
resolves the CCCL headers and the CUDA toolkit. The headers come from a
configured root, the checkout that contains the package, or the installed
header bundle.

Primitive families
------------------

A :term:`family` groups related primitives and their implementation. Scan
includes ``scan``, ``inclusive_scan``, ``exclusive_scan``, ``inclusive_sum``,
and ``exclusive_sum``. Its public entry points and type declarations live in
``_group_scan.py`` and ``_group_scan.pyi``. Planning lives in
``_planning/group/scan.py`` and CuTe adaptation in ``_lowering/_scan.py``.
Paths in this guide that begin with an underscore are relative to
``python/cuda_coop_cutlass/cuda/coop/cutlass``.

.. _from-a-call-to-a-typed-provider-request:

Planning the CUB call
---------------------

For Load, ``_group_load_store.py`` first creates the result payload and
checks the group, algorithm, and controls. It classifies a control as
omitted, a compile-time constant, or a runtime value. The lowering then
reads the CuTe memory operand's element type, checks a ``dtype`` the caller
declared against it, and checks that the operand's layout exposes the
contiguous pointer required by CUB.

The planner uses those arguments and the kernel's launch dimensions to
choose a CUB specialization. It also determines which threads must participate
and what scratch storage and barriers the call needs. The lowering puts the
plan and CuTe scalar type into a provider request, which the renderer uses
to generate the wrapper.

In the example, the block group and ``algorithm="transpose"`` select the
transpose form of CUB Block Load and Store for a 64-thread block with two
integers per thread. Thread rank ``t`` owns tile items ``2 * t`` and
``2 * t + 1``. Block Scan computes exclusive sums in that blocked order. The
generated wrappers and scratch plans reflect each primitive's algorithm,
element type, and items per thread.

The renderer generates a device wrapper with an ordinary callable symbol.
CuTe's ``cute.ffi`` emits the call to that symbol. Load uses a result buffer
to return its items, then updates the ``ThreadData`` values after the FFI
call is emitted. Store passes the existing items to its
provider. The plan identifies equivalent requests so they can share one
generated function and cached artifact.

Other primitive families use the same approach. For Reduce, the planner
selects CUB BlockReduce or WarpReduce, checks scalar or per-thread payload
support, and records scratch requirements. Block reductions accept an explicit
``TempStorage`` descriptor. Warp reductions use compiler-owned storage.
The Reduce lowering generates the wrapper and adapts CuTe's values to its
arguments. The returned scalar is defined only at group rank zero.

.. _coop-cutlass-exact-launch-facts:

Exact launch facts
-------------------

``_compiler/_launch.py`` reads the compiler-provided exact block,
grid, and cluster dimensions and the cooperative/cluster launch flags. The
adapter records their origin in the ``LaunchFacts`` object. If a
dimension or flag is unavailable, it stays unknown in that object. Planning
fails if the primitive needs the missing value.

For the prefix sum, block dimensions determine both the CUB specialization
and linear rank: ``x + block_x * (y + block_y * z)``. Warp primitives also
need the exact block size to establish complete physical warps and allocate
one scratch slice per group. Cluster and grid queries and cluster ``sync``
need consistent cluster dimensions and launch mode.

Do not substitute ``maxntid`` for exact dimensions: an upper bound does not
prove the number of participating threads. The adapter takes every launch
fact from ``cute._get_launch_facts()``. It ignores Python frames and
user-maintained launch descriptions.

One provider bundle per trace
------------------------------

Each trace that makes a cooperative call has a provider session keyed by
its compile options and MLIR module. The session deduplicates equivalent
requests and records deferred scratch uses. Load, Scan, and Store in the
example contribute to the same bundle. Finalization compiles the three
providers together.

``cuda.coop.cutlass`` registers one persistent trace-finalization hook with
CuTe. The hook runs for every trace on that DSL. It finds the session
belonging to the module being finalized and leaves unrelated or nested
sessions alone. The hook renders the requests into one C++ translation unit
and compiles it to NVIDIA LTO-IR. It marks the provider calls convergent
and attaches the artifact to each ``gpu.module`` in the trace through its
``link-libraries`` attribute. An operator from ``compile_operator`` that a
request uses has its own LTO-IR, which the hook attaches the same way. The
CuTe compiler and final linker then resolve the provider calls while
compiling the enclosing kernel.

Header and toolkit discovery use the helpers under ``_headers``.
The compiler libraries for the selected toolkit are loaded before importing
the NVRTC bindings, and the loaded NVRTC version is checked against that
selection. A mismatch reports an error before provider compilation. The
final linker must also accept the generated LTO-IR. A bundle that NVRTC
compiles can still fail at the final link.

The cache identity includes the rendered source, header identity, compiler
options, architecture, and resolved toolkit/compiler information. Cache
loads check the artifact's size and digest and any recorded scratch
layouts. Cache writes use a lock and atomic replacement. The LTO-IR file
remains available until the final linker has read it. Finalization removes
stale managed library paths from persistent compiler options so the next
trace cannot link an earlier bundle by mistake.

Failed emission restores the queued requests and pending scratch uses saved in
the provider-session snapshot. Emitted IR and payload changes stay in place.
Finalization removes only its own session. Lifecycle tests exercise failed
compilation or linking followed by a successful retry, as well as repeated and
nested compilation. These checks matter because one Python process can compile
many kernels through the same CuTe DSL.

.. _coop-cutlass-scratch-allocation:

Scratch allocation after tracing
---------------------------------

Direct, striped, and vectorized Load/Store do not use scratch. Their provider
ABI omits a storage pointer. Passing a ``TempStorage`` descriptor adds no
reuse barrier and leaves the storage-free provider's identity unchanged.

Calls with deferred scratch record its use while tracing. The finalizer
adds ``sizeof`` and ``alignof`` probes to the same NVRTC program as the
provider bundle. The resulting C++ layouts determine the allocation. The
size each of these calls needs comes only from its probe. Other
providers, including Warp Scan, declare their shared storage inside the
generated C++ wrapper, so they skip this placeholder-replacement step.

Deferred events let one explicit descriptor serve several calls. Shared
storage uses enough capacity for the largest requirement, while exclusive
storage assigns separate slices to distinct call sites. Requested alignment
is a minimum. An explicit capacity that cannot accommodate its uses is an
error. Once the layouts are known, the finalizer materializes shared-memory
allocations and replaces the trace's storage placeholders. It then raises
the shared-memory size of each traced launch of a kernel that received
scratch to at least the kernel's total, which ``cute.kernel_smem_size``
reports. A launch that passes its own ``smem`` size therefore has room for
the scratch.

``TempStorage.reserve`` records an event with a known byte size and
alignment, and returns an ordinary CuTe tensor built from an address
placeholder. It participates in the same descriptor layout as primitive
scratch. Finalization resolves these views even when the kernel has no
cooperative provider calls. A descriptor with reservations requires
``auto_sync=False`` for every use. Finalization also checks earlier and later
primitive events before materializing its allocation.

``TempStorage.release(dtype=None)`` consumes the descriptor and returns its
complete region as a one-dimensional native CuTe tensor. The default dtype
is ``cutlass.Uint8``. Inferred size resolves during finalization. Exclusive
slices, their padding, and explicit extra capacity are included. The region
must be nonempty, its size divisible by the element size, and its guaranteed
alignment sufficient for the requested type. The descriptor must be
constructed inside the kernel.
For a typed release without preceding primitive or reservation uses, specify
``alignment`` of at least the dtype's element size on the constructor.

Release adds no allocation or synchronization. Complete previous accesses
and asynchronous operations before using overlapping bytes. All aliases of
the descriptor become unusable on the executing path, including calls to
``reserve``, ``sync``, ``release``, and storage-free primitives. The finalizer
checks reachable branch and loop paths, so a conditional release prevents a
later use after the branches join. Mutually exclusive releases are valid,
as is a fresh descriptor constructed and consumed within each iteration.
Runtime selection between different descriptor instances is rejected at
every use of the descriptor. Without that check, the call would use the
Python trace's template for either instance. Identity tokens and use
markers support this analysis without device-side lifetime flags.

Exclusive slices still need reuse synchronization when one call site runs
again in a loop. Explicit descriptors default to ``auto_sync=False`` for both
sharing modes, so the kernel must call ``storage.sync()`` before reuse. Set
``auto_sync=True`` to insert automatic trailing synchronization.
Compiler-managed scratch synchronizes automatically. Physical and logical
warp primitives use independent per-group storage and a warp barrier masked
to the group's lanes. See the :doc:`API reference <api>` for each family's
explicit-storage support and participation rules.

Finding the implementation
---------------------------

Paths below are relative to ``python/cuda_coop_cutlass/cuda/coop/cutlass``.
Planning describes primitive semantics. Lowering adapts CuTe values and emits
provider calls.

.. list-table:: Source map
   :header-rows: 1
   :widths: 36 64

   * - Area
     - Starting points
   * - Header and toolkit discovery
     - ``_headers/__init__.py``, ``_headers/_identity.py``,
       ``_headers/_toolkit.py``
   * - Compiler validation and launch facts
     - ``_compiler/_runtime.py``, ``_compiler/_launch.py``
   * - User-facing values and groups
     - ``_thread_data.py``, ``_thread_group.py``, ``_temp_storage.py``,
       and their ``.pyi`` files
   * - Family planning and lowering
     - ``_planning/group/`` and ``_lowering/``. Public entry points live
       in the corresponding ``_group_*.py`` modules.
   * - Provider sessions and finalization
     - ``_compiler/_state.py``, ``_compiler/_finalize.py``,
       ``_compiler/_convergence.py``
   * - Rendering, compilation, and artifacts
     - ``_compiler/_rendering.py``, ``_compiler/_bundle.py``,
       ``_compiler/_nvrtc.py``, ``_compiler/_cache.py``
   * - Scratch layout, materialization, and lifetime
     - ``_compiler/_layout.py``, ``_compiler/_storage.py``,
       ``_compiler/_lifetime.py``

Keeping documentation in sync
------------------------------

When changing an API or compiler path, update its documentation in the same
change. The diagrams are SVG source files, and the build copies them as
they are. A change to code that a diagram shows needs a matching edit to
that diagram.

* Each rule is stated once, in the docstring of the function, class or
  module it governs. ``concepts.rst``, the Programming Guide, Advanced
  topics, the FAQs and the visualization pages link to that entry in the
  API reference. When participation, layout, result ownership, or storage
  rules change, update that docstring, then check the pages that link to
  it. Keep the executable examples in
  ``python/cuda_coop_cutlass/examples/`` and the runtime tests in
  ``python/cuda_coop_cutlass/tests/runtime/`` consistent with those rules.
  Preserve the markers used by ``literalinclude``.
* Every complete kernel shown in a user page or docstring comes from a
  tested file through ``literalinclude``. The kernel and launcher in
  ``visualizations/index.rst`` hold no primitive call and are written in
  the page. An include of part of a file
  names ``:start-after:`` and ``:end-before:`` markers, and each marker
  occurs once in that file. Mark a new region in an example under
  ``python/cuda_coop_cutlass/examples/`` with ``# docs: start <name>`` and
  ``# docs: end <name>``. Mark a new region in a test under
  ``python/cuda_coop_cutlass/tests/runtime/`` with
  ``# cutlass-<name>-begin`` and ``# cutlass-<name>-end``. A docstring
  writes its include paths relative to ``docs/python/coop_cutlass/``, where
  ``api.rst`` renders it, and has a GitHub link to the included file above
  the directive.
* ``python/cuda_coop_cutlass/tests/test_docs_sources.py`` checks the file
  and markers of every include, the GitHub link above each include in a
  docstring, and the ``currentmodule`` line after each ``automodule`` in
  ``api.rst``. It reads the sources as text and needs no GPU. Run it from
  the repository root after changing a marker, an include or ``api.rst``:

  .. code-block:: bash

     python -m pytest -q python/cuda_coop_cutlass/tests/test_docs_sources.py

  ``docs/conf.py`` suppresses docutils warnings, so the Sphinx build passes
  when an include finds no file or marker and leaves the snippet out of the
  page. CI runs this test when a pull request changes
  ``python/cuda_coop_cutlass/``. CI skips it for a pull request that
  changes only files under ``docs/``, so run it by hand.
* For compiler changes, check the source map above and the phase and swimlane
  diagrams in ``docs/python/coop_cutlass/visualizations/``. Update both the
  overview and Python-class SVGs, including each ``-mobile.svg`` variant.
  Check class names, call order, ownership, and linking boundaries against
  the implementation.
* ``test_debugger_walkthrough.py`` and
  ``test_cutlass_integration_example.py`` in
  ``python/cuda_coop_cutlass/tests/runtime/`` run
  ``debugger_walkthrough.py`` and ``cutlass_integration_example.py`` from
  ``docs/python/coop_cutlass/``. Both tests need a GPU. CI skips them for a
  pull request that changes only files under ``docs/``, so run them by hand
  after changing either script. ``cutlass-integration-example.svg`` names
  the functions and handoffs of the second script, so update it when they
  change.
* ``python/cuda_coop_cutlass/examples/tensor_memory_topk.py`` runs only on
  an SM100 or SM103 GPU. Its test, ``test_tensor_memory_topk.py`` in
  ``python/cuda_coop_cutlass/tests/runtime/``, skips on other GPUs, CI's
  example loop leaves the file out, and CI runs this package on L4 and H100
  GPUs. Run the test by hand on an SM100 or SM103 GPU after changing the
  example or its page, :doc:`tensor_memory_topk`.
* For operation visualizations, update the RST explanation, static SVG, and
  corresponding ``docs/_static/coop-*.js`` explorer together. A family's
  module docstring can show the same SVG. Its ``:alt:`` text and caption
  quote the picture's numbers, so update them with the SVG. The mapping in
  ``docs/_ext/coop_visualization.py`` connects API entries to these pages.
  The build adds a link to the page under each mapped entry, so a docstring
  carries no such link. Add a new public function to
  ``_API_VISUALIZATIONS`` and a new page to ``_EXPLORERS``. Add the page to
  ``_VISUALIZATION_TITLES`` when its link title differs from its
  capitalized name, as ``TopK`` and ``Merge Sort`` do. The opening of
  ``visualizations/index.rst`` says once what every picture simplifies. A
  page says only what is special about its own picture.
* ``concept-temp-storage.svg`` is drawn to scale from scratch sizes measured
  for the ``prefix_sum_shared``, ``prefix_sum_exclusive`` and
  ``prefix_sum_omitted`` kernels in
  ``python/cuda_coop_cutlass/tests/runtime/test_concepts_documentation.py``:
  128 threads, four ``Int32`` items each, CUB from CCCL 3.7. The same sizes
  appear in the ``_temp_storage`` module docstring and in the figure's
  caption in ``concepts.rst``. When the planner's layout or CUB's scratch
  sizes change, measure again and update all three.
  ``cute.arch.dynamic_smem_size()`` inside each kernel gives the totals, and
  releasing one ``TempStorage`` per call gives each call's size.
* ``concept-thread-groups.svg`` draws the ``group_sums`` kernel in that test
  file: a block of 128 threads and its groups of 32 and 8.
  ``concept-thread-data.svg`` draws the ``prefix_sum`` example: a block of
  128 threads with four ``Int32`` items per thread. Each appears in
  ``concepts.rst`` and in a module docstring (``_thread_group`` and
  ``_thread_data``). When either kernel's block size, groups or item count
  changes, redraw the figure and update the caption and ``:alt:`` text in
  both places.
* Module docstrings that ``api.rst`` includes with ``automodule`` are user
  documentation: ``__init__``, ``_thread_group``, ``_thread_data``,
  ``_temp_storage``, ``_group_load_store``, ``_group_reduce``,
  ``_group_reduce_batched``, ``_group_scan``, ``_group_exchange``,
  ``_group_shuffle``, ``_group_merge_sort``, ``_group_radix_sort``,
  ``_group_topk``, ``_group_neighbors``, ``_group_histogram``,
  ``_group_run_length`` and ``_device_operator``. ``api.rst`` renders each
  under its section heading with the options ``:no-members:``,
  ``:no-special-members:`` and ``:no-index:``, followed by
  ``.. currentmodule:: cuda.coop.cutlass``. Notes for maintainers go in an
  ``# Implementation notes.`` comment below the docstring.
* When a CUB block or warp primitive under ``cub/cub/block`` or
  ``cub/cub/warp`` changes, or a new one is added, update this package, its
  tests and its documentation in the same pull request.
* User pages and public docstrings follow the prose rules in
  ``python/cuda_coop_cutlass/AGENTS.md``. This guide, the primer and the
  integration example follow them too, and may use the implementation's
  words: provider, lowering, planner, wrapper, descriptor, backend, adapter
  and trace time.

Build the documentation with ``docs/gen_docs.bash``, which treats Sphinx
warnings as errors. Run ``test_docs_sources.py`` and the affected examples
or runtime tests. Inspect the rendered pages, source links, and diagrams at
desktop and narrow widths. Check the static fallback when an interactive
visualization changes.

.. _cuda.coop.cutlass.debugger_walkthrough:

Debugger Walkthrough
--------------------

Follow a tile copy through a Python debugger to see the planner
choose a CUB specialization, CuTe emit the device calls, and NVRTC compile
their providers. These stops are in the host Python code doing the
compilation. The GPU executes the resulting kernel later. A Python
breakpoint does not stop a GPU thread.

An example to debug
^^^^^^^^^^^^^^^^^^^

Open ``docs/python/coop_cutlass/debugger_walkthrough.py`` in your checkout,
or `view the example on GitHub
<https://github.com/NVIDIA/cccl/blob/main/docs/python/coop_cutlass/debugger_walkthrough.py>`_.
It needs no command-line arguments. Its kernel and launcher are:

.. literalinclude:: debugger_walkthrough.py
   :language: python
   :start-after: docs: start cutlass-debug-kernel
   :end-before: docs: end cutlass-debug-kernel

The default ``algorithm`` is ``"direct"`` and ``items_per_thread`` is ``2``.
Each of the 128 threads copies two integers, so the default tile contains 256
items. The kernel and launcher specialize the item-count argument with
``cutlass.Constexpr``. Pass ``--items-per-thread`` to change it. ``scratch``
is a descriptor. Direct Load and Store use no scratch, so no shared memory
is allocated for it. The same descriptor shows scratch reuse in the
transpose pass.

The complete example allocates device buffers with the CUDA Driver API,
wraps their addresses in CuTe pointers, and frees them before returning.
It compiles the launcher once, then calls the resulting function twice:

.. literalinclude:: debugger_walkthrough.py
   :language: python
   :start-after: docs: start cutlass-debug-launches
   :end-before: docs: end cutlass-debug-launches

Each iteration resets the device output to ``-1``, launches the copy,
synchronizes, and checks every value against NumPy. Here,
``cute.compile`` makes compilation an explicit step before either launch.
The two calls to ``compiled`` reuse that kernel.

A direct call to a CuTe ``@cute.jit`` function traces again to compute the
module's cache key, even when compiled code is reusable. The example
keeps the compiled callable, so both launches skip that trace.

Configure VS Code
^^^^^^^^^^^^^^^^^

Open this CCCL checkout as the VS Code folder. Use **Python: Select
Interpreter** to select an environment satisfying the
:ref:`CUTLASS compiler requirements <coop-cutlass-compiler-requirements>`
and the :ref:`installation requirements <coop-cutlass-requirements>`.
A CUTLASS package that imports successfully may still lack an interface
needed by this walkthrough. Install the Python and Python Debugger
extensions where the program runs, including the remote side when using
Remote SSH.

Add this configuration to ``.vscode/launch.json``. If the file already
exists, add the entry to its ``configurations`` list:

.. code-block:: json

   {
     "version": "0.2.0",
     "configurations": [
       {
         "name": "cuda.coop.cutlass: Debug active Python file",
         "type": "debugpy",
         "request": "launch",
         "program": "${file}",
         "console": "integratedTerminal",
         "cwd": "${workspaceFolder}",
         "justMyCode": false,
         "env": {
           "PYTHONPATH": "${workspaceFolder}/python/cuda_coop_cutlass",
           "CUDA_COOP_CUTLASS_CCCL_ROOT": "${workspaceFolder}",
           "CUDA_COOP_CUTLASS_SOURCE_DUMP_DIR": "${workspaceFolder}/build/coop-cutlass-debug-sources"
         }
       }
     ]
   }

``justMyCode: false`` permits stepping into library code. ``PYTHONPATH`` and
``CUDA_COOP_CUTLASS_CCCL_ROOT`` select this checkout's Python sources and
C++ headers. ``cute.compile`` explicitly requests CuTe compilation. When run
as a script, the example also selects a new temporary provider-cache
directory and removes it on exit. A new process with that empty provider
cache lets each debug session reach NVRTC.

On a machine with several GPUs, add ``CUDA_VISIBLE_DEVICES`` to ``env``.
Use the toolkit configured for your compatible compiler environment.

Set the first breakpoint on ``import cuda.coop.cutlass as coop`` in the
example. Keep that file active, select the named configuration in
**Run and Debug**, and press **F5**. ``${file}`` selects the active editor
file when the session starts. **F10** steps over, **F11** steps into,
**Shift+F11** steps out, and **F5** continues. Debug Console expressions use
the selected Call Stack frame. See the
`VS Code Python debugging documentation
<https://code.visualstudio.com/docs/python/debugging>`_ for those controls.

The stops below identify executable statements by function name, so they
remain useful as line numbers change. Set each breakpoint before continuing
past the stage that calls it. Avoid calling primitives from Debug Console:
they can emit additional IR into the paused trace.

Initial import and compiler validation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The first stop is on ``import cuda.coop.cutlass as coop``, before that
line runs. The example has already imported CuTe. In
``_compiler/_runtime.py``, set a breakpoint in ``validate_cutlass_runtime``
before continuing through the package import. Step over the
``missing_capabilities`` assignment and inspect it. It is empty when the
installed CuTe has the ``CuTeDSL`` accessor and trace-finalization hooks,
the compiler-environment and launch-facts queries, and the link-library and
architecture options that this package uses. The function checks that
these interfaces exist. Compilation and device execution come later.

Disable the import breakpoints and stop on
``compiled = cute.compile(...)`` in the example. Evaluate:

.. code-block:: python

   coop.__file__
   cutlass.__file__
   source[:8]
   destination[:8]
   algorithm

``coop.__file__`` should point inside your checkout. ``cutlass.__file__``
identifies the dependency selected by the interpreter. The arrays start
with ``0, 1, ..., 7`` and eight ``-1`` values respectively, and
``algorithm`` is ``"direct"``. Compilation and both launches are still
ahead of you.

Follow the launch dimensions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Continue from ``cute.compile`` until CuTe traces the kernel. Cooperative
calls execute directly in this trace and read its exact launch dimensions.
Paths for the following stops are relative to
``python/cuda_coop_cutlass/cuda/coop/cutlass/``.

In ``_group_load_store.py:_resolve_group()``, break on
``resolved = _resolve_primitive_group_from_launch(...)``. At that stop,
inspect:

.. code-block:: python

   operation
   launch.exact_block_dim
   launch.exact_grid_dim
   launch.provenance

The first operation is ``"load"``, with block ``(128, 1, 1)`` and grid
``(1, 1, 1)``. The provenance names the compiler's launch-facts API.
To follow the call into CuTe, step into
``current_kernel_launch_facts()`` in a new debug session. These dimensions
come from ``.launch(grid=1, block=128)``. The block dimensions determine the
CUB specialization.

From a primitive call to a plan
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set a breakpoint in ``_lowering/_load_store.py:_make_request()`` on
``return _CubLoadStoreRequest(plan, value_type)``. Continue and inspect:

.. code-block:: python

   kind
   value_type
   plan.target
   dict(plan.implementation.template_arguments)
   plan.participation
   plan.temp_storage
   plan.synchronization

The plan selects ``CUB_BLOCK``, ``cutlass.Int32``, 128 threads, two items
per thread, and ``::cub::BLOCK_LOAD_DIRECT``. Participation requires
the full block. Storage ownership and the storage-reuse barrier are
``NONE``, even though the example supplied ``scratch``.

Set the next breakpoint in ``provider_load()`` in the same file, on
``_state.register_request(request)``. Inspect ``request.cpp_type``,
``request.symbol_name``, ``runtime_types``, and ``runtime_args``. The C++
type specializes ``cub::BlockLoad``. The symbol names the device wrapper
that will implement it. Both runtime-control lists are empty, because
this copy passes no ``valid_items``, ``offset``, or ``oob_default``.

Step over request registration. It adds the request to the session owned
by this trace. Constructing ``scratch`` already created that session and
registered the finalization hook. No C++ has been compiled yet.

Emit the device calls
^^^^^^^^^^^^^^^^^^^^^

Still in ``provider_load()``, stop on ``ffi(...)`` after
``scratch_types, scratch_args = _scratch_arguments(...)``. Both are empty
tuples. The FFI signature has two pointers: the source address
and a per-thread result buffer containing two integers.

Step over the complete FFI call and the following payload assignments.
CuTe has emitted a call into its IR, and ``output`` now holds the CuTe
values read from that result buffer. These are compiler values. The GPU
loads each thread's integers while the kernel runs.

Continue. The launch-dimensions breakpoint stops first, with
``operation`` equal to ``"store"``. Continue again to the plan breakpoint
for Store. It selects ``::cub::BLOCK_STORE_DIRECT``. In
``provider_store()``, break on its ``ffi(...)`` call and inspect ``values``
and ``request.symbol_name``. Store passes the destination pointer and the
two item values. Unlike Load, it needs no result buffer. Disable the plan
and emission breakpoints after inspecting both operations.

Finalize the trace and compile the providers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before continuing from Store, set a breakpoint in
``_compiler/_finalize.py:_trace_finalize_hook()`` on
``source = _rendering.render_bundle_source(requests)``. This statement
comes after the checks that select this module's provider session, so
unrelated modules do not stop here. Inspect:

.. code-block:: python

   len(requests)
   [request.symbol_name for request in requests]
   session.deferred_temp_storage_event_list()
   print(module)

There are two requests, one for Load and one for Store, and no deferred
storage events. The IR contains calls to the symbols you saw in the
lowerings. Step over the rendering assignment and inspect ``source``.
Both wrappers belong to one C++ translation unit. The Load wrapper builds
a two-item C++ array, calls ``BlockLoad::Load``, and writes the result
buffer. The Store wrapper builds its array from the two scalar arguments
and calls ``BlockStore::Store``.

In ``_compiler/_bundle.py:_compile_bundle_source()``, break on
``options = _nvrtc.compiler_options(context, arch)``. Continue, inspect
``context.include_dirs`` and ``context.toolkit_root``, then step over
the assignment. ``options`` contains C++17, relocatable device code,
``-dlto``, the target architecture, and the include directories. This is
also a useful stop for detecting headers or toolkit libraries from the
wrong installation.

Before continuing, set a breakpoint in ``_compiler/_nvrtc.py:_compile_ltoir()``
on ``error, program = nvrtc.nvrtcCreateProgram(...)``. At that stop,
``source`` contains both providers and ``prepared`` is ``None`` for this
storage-free copy. Step through ``nvrtcCompileProgram`` and its error
check, then the ``nvrtcGetLTOIRSize`` and ``nvrtcGetLTOIR`` calls. ``blob``
contains the provider LTO-IR. The full C++ is also saved under
``build/coop-cutlass-debug-sources`` for inspection outside the debugger.

Hand the link input back to CuTe
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set a breakpoint near the end of ``_trace_finalize_hook()`` on
``_bundle.append_link_library_attr(module, path)``. Continue and inspect
``path``. It names the cached ``.ltoir`` file containing both providers.
The statement before this one,
``_convergence.lower_convergent_provider_calls(module, requests)``, has
already run. ``print(module)`` now shows the Load and Store calls as
``llvm.call`` operations with the ``convergent`` attribute, where the
rendering stop showed ``func.call``. Step over
``append_link_library_attr`` and evaluate:

.. code-block:: python

   [(op.name, str(op.attributes["link-libraries"]))
    for op in module.body.operations if op.name == "gpu.module"]

The GPU module now names that file as a link input. CuTe's compiler and
final linker consume it to resolve the device calls. The example keeps
its temporary cache alive through compilation and both launches.
This handoff is a useful boundary when diagnosing a failure. When you reach
it, provider compilation has finished. Kernel linking and execution are
still ahead.

Finish the launch and observe reuse
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Disable the compiler breakpoints and set one on
``compiled(source_pointer, destination_pointer)`` in the example.
Continue. ``cute.compile`` has returned a callable. The first launch is
about to execute it with ``iteration == 0``. Continue again to reach
the second launch with ``iteration == 1``. The first verification message
has printed, and ``destination`` has been reset to ``-1`` on both host
and device.

Re-enable the plan and NVRTC breakpoints, then continue. Calling ``compiled``
executes the existing kernel without tracing or provider compilation. Neither
breakpoint should fire, and the second verification message should print.
Calling ``cute.compile`` again would request another CuTe compilation, though
the provider cache could supply its LTO-IR. Start a new debug session to
repeat the tour. Editing a file while paused does not replace the function
already loaded in the current process.

A second pass: shared scratch and synchronization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Add ``"args": ["--algorithm", "transpose"]`` to the launch configuration
and start a new debug session. From a terminal, the equivalent is:

.. code-block:: bash

   python docs/python/coop_cutlass/debugger_walkthrough.py --algorithm transpose

The kernel, launch dimensions, and expected output are unchanged. At the
plan stop, ``ALGORITHM`` now selects ``::cub::BLOCK_LOAD_TRANSPOSE`` or
``::cub::BLOCK_STORE_TRANSPOSE``. Storage ownership is ``CALLER``, and the
storage-reuse barrier is ``BLOCK``. The FFI signatures gain three scratch
arguments: a shared-memory address, its capacity in bytes, and the
automatic-synchronization flag.

Set these additional breakpoints before reaching Load:

* In ``_compiler/_storage.py:register_deferred_temp_storage_event()``,
  break on ``session.add_deferred_temp_storage_event(...)``. Inspect
  ``primitive_name``, ``requirement_key``, ``smem_addr_placeholder``, and
  ``size_placeholder``. The address and size are fresh IR placeholders.
  NVRTC has not yet supplied the C++ scratch layout. Load and Store record
  separate uses of the same ``TempStorage`` descriptor.
* In ``_compiler/_finalize.py:_trace_finalize_hook()``, break on the second
  ``plans = _storage.plan_deferred_temp_storage_events(...)``. Inspect
  ``compilation.layouts``. NVRTC has compiled both providers and recovered
  their exact C++ ``sizeof`` and ``alignof`` requirements from the same
  program. Step over planning and inspect ``plans``: there is one shared
  allocation, sized for the larger requirement. Its two bindings have
  byte offset zero. The size and alignment come from this build. Another
  toolkit can give different numbers.
* In ``_compiler/_storage.py:materialize_deferred_temp_storage_plans()``,
  break on ``smem_ptr = allocator.allocate(...)``. Inspect
  ``plan.size_in_bytes``, ``plan.alignment``, and ``plan.bindings``. Step
  through allocation and operand replacement to see the recorded calls
  acquire their actual address and capacity.

At the link-input stop, compare ``print(module)`` with the Direct version.
There is now a shared allocation and the calls carry scratch operands.
Open the dumped C++ and find ``temp_storage_auto_sync``. Each wrapper
ends with a conditional ``__syncthreads()``. This example passes a true
flag, so Load finishes using the workspace before Store reuses it.

``cuda.coop.cutlass`` emits the trailing barrier in the provider wrapper
and patches deferred storage operands after obtaining the C++ layouts.
``ThreadData`` still holds the per-thread payload. ``TempStorage`` describes
the shared workspace used during each primitive. Continue to verify both
launches of the transpose copy.

If a breakpoint does not stop
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Check ``coop.__file__`` at the first launch. The executing module and
  breakpoint must belong to the same checkout. Inspect ``cutlass.__file__``
  too if a required compiler interface is missing.
* Use the named launch configuration with ``justMyCode: false``. A hollow
  breakpoint can remain pending until its module loads. Put it on an
  executable statement rather than a decorator or ``def`` line.
* If the stop in ``validate_cutlass_runtime`` is skipped, the package was
  already imported when you set the breakpoint. Validation runs once, on
  the first ``import cuda.coop.cutlass``, whether CuTe or the package is
  imported first. A failed validation raises an import error. Read it
  before investigating later compiler stops.
* If NVRTC is skipped, restart the complete script. Its temporary provider
  cache is created only under ``if __name__ == "__main__"``. Check that
  the provider directory is new and that you set the breakpoint before
  ``cute.compile``. A stop in ``_compile_bundle_source()`` can show a
  provider-cache hit. The backend writes the source dump before it
  looks up a cached provider, so a dump appears whether or not NVRTC ran.
* If a later launch retraces, check that you are calling ``compiled``.
  This walkthrough retains the callable returned by ``cute.compile``
  and invokes it twice. Calling the decorated ``launch`` directly follows
  CuTe's JIT path instead.

Working on a primitive
----------------------

Start with the C++ overload and a small kernel exercising it. Check the
family plan for payload layout, participation, result ownership, and storage
requirements. Keep the public entry point and its ``.pyi`` declaration aligned.

Adapt the plan in ``_lowering``. The lowering must agree with its provider
wrapper about argument types, result buffers, and scratch operands. Register
the typed request with the current session and preserve rollback if emission
fails. Load demonstrates a newly returned payload. Scan and Reduce show
additional result contracts and controls. Reuse the existing renderer
registration, bundle finalizer, and storage planner when their contracts
express the operation.

Checking a change
-----------------

Start with the affected family under
``python/cuda_coop_cutlass/tests``. Unit tests check plan selection,
argument contracts, and state transitions. Compile tests exercise real CuTe
traces, launch metadata, FFI emission, and lifecycle recovery. Runtime tests
compare results and layouts against independent references, including
partial tiles and repeated scratch reuse. For storage-policy routing, see
``unit/test_load_store_plans.py``. For convergent provider calls, see
``compile/test_convergence.py`` and ``runtime/test_convergence.py``.

For a Load/Store or lifecycle change, the following are useful entry points
from the repository root with the package and compatible compiler installed:

.. code-block:: bash

   python -m pytest -q python/cuda_coop_cutlass/tests/unit/test_compiler.py
   python -m pytest -q python/cuda_coop_cutlass/tests/compile/test_compiler_lifecycle.py
   python -m pytest -q python/cuda_coop_cutlass/tests/runtime/test_block_load_store.py
   python -m pytest -q python/cuda_coop_cutlass/tests/runtime/test_block_algorithms.py

The compile and runtime examples above require a compatible CUDA environment
and GPU. Inspect skips when assessing coverage. A final-cubin check skips
when ``cuobjdump`` is missing, and a skipped check has tested nothing.
Fresh-process and installed-artifact tests check namespace coexistence,
module origins, import diagnostics, and typing. Run affected planning tests
when changing specialization or participation rules.

To inspect the generated C++, set ``CUDA_COOP_CUTLASS_SOURCE_DUMP_DIR`` before
compiling a kernel. The dump includes the source on provider-cache hits too.
``CUDA_COOP_CUTLASS_PROVIDER_CACHE_DIR`` selects the provider artifact cache
when a test needs an isolated cache directory. See :doc:`configuration`
for these settings and header-selection controls.

For generated-code claims, inspect the final linked cubin with ``cuobjdump``.
``runtime/test_block_load_store.py`` checks that direct Load/Store leave no
provider calls, shared accesses, or barriers. The final-cubin check in
``runtime/test_block_algorithms.py`` tests striped, vectorize, and transpose.
It requires shared-memory accesses (``LDS`` and ``STS``) and barriers only
for transpose. The final-cubin
checks in ``runtime/test_warp_load_store.py`` and
``runtime/test_logical_warp_load_store.py`` reject provider calls and block
barriers for physical and logical Warp operations. They forbid shared memory
and warp synchronization for storage-free algorithms. Transpose must read
and write shared memory. For the width-one logical-Warp case, shared memory
is not checked.

The final-cubin check in ``runtime/test_reduce.py`` covers a block sum with
the default algorithm and with ``raking``. It rejects leftover provider
symbols, ``CALL`` instructions and generic ``LD`` and ``ST`` accesses, and it
requires shared-memory ``LDS`` and ``STS``. It does not check warp
reductions, shared-memory size, registers, or barriers.

The final-cubin check in ``runtime/test_scan.py`` rejects provider calls for
block and logical-Warp Scan, and block barriers for the logical-Warp path.
It also rejects generic ``LD`` and ``ST`` accesses and requires ``LDS`` and
``STS`` on the block path. It saves resource reports but does not check
them.

The final-cubin checks in ``runtime/test_exchange.py`` and
``runtime/test_shuffle.py`` reject provider calls for block and logical-Warp
Exchange and for array and scalar Shuffle. The Exchange check also rejects
block barriers on the logical-Warp path. Neither check examines shared
memory. ``runtime/test_group_hierarchy.py`` checks that mapped-group queries
leave no calls, barriers, warp synchronization, or shared memory.

The final-cubin check in ``runtime/test_merge_sort.py`` rejects provider
calls for partial block and logical-Warp sorts. It also rejects block
barriers on the logical-Warp path. It does not check shared memory.

Provider source and intermediate PTX come from earlier stages, so check the
linked cubin for the final result. Use Compute Sanitizer race checks for
changes to scratch allocation or synchronization. Numerical correctness,
generated-code evidence, and public-package qualification require separate
checks.
