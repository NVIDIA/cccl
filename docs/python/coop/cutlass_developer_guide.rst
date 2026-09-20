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

For a hands-on tour, follow the :ref:`cuda.coop.cutlass.debugger_walkthrough`.
Both compiler walkthroughs use a 128-thread tile copy, so their registration,
planning, code generation, and linking steps can be compared directly.

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
private ``_compiler_scope`` takes precedence over environment detection.
Both integrations can be registered in the same process. If two predicates
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
participate and what scratch storage and barriers the call needs. Both
backends use this planner. The CUTLASS lowering puts the plan and CuTe scalar type
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

.. _coop-cutlass-exact-launch-facts:

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

.. _coop-cutlass-scratch-allocation:

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
       Shuffle, Merge Sort, Radix Sort/Rank, and TopK files follow the same
       organization
   * - Launch facts and provider sessions
     - ``cutlass/_compiler/_launch.py``, ``cutlass/_compiler/_state.py``,
       ``cutlass/_compiler/_finalize.py``
   * - Rendering, compilation, and artifacts
     - ``cutlass/_compiler/_rendering.py``, ``cutlass/_compiler/_bundle.py``,
       ``cutlass/_compiler/_nvrtc.py``, ``cutlass/_compiler/_cache.py``
   * - Scratch layout and materialization
     - ``cutlass/_compiler/_layout.py``, ``cutlass/_compiler/_storage.py``

.. _cuda.coop.cutlass.debugger_walkthrough:

Debugger Walkthrough
--------------------

Follow a tile copy through a Python debugger to see the shared planner
choose a CUB specialization, CuTe emit the device calls, and NVRTC compile
their providers. These stops are in the host Python code doing the
compilation. The GPU executes the resulting kernel later; a Python
breakpoint does not stop a GPU thread.

An example to debug
^^^^^^^^^^^^^^^^^^^

Open ``docs/python/coop/cutlass_debugger_walkthrough.py`` in your checkout,
or :download:`download the example <cutlass_debugger_walkthrough.py>`.
It needs no command-line arguments. Its kernel and launcher are:

.. literalinclude:: cutlass_debugger_walkthrough.py
   :language: python
   :start-after: docs: start cutlass-debug-kernel
   :end-before: docs: end cutlass-debug-kernel

The default ``algorithm`` is ``"direct"``. Each of the 128 threads copies
two integers, so the complete tile contains 256 items. ``scratch`` is a
descriptor: Direct Load and Store ignore it and allocate no shared memory.
The same descriptor will let us follow scratch reuse in the transpose pass.

The complete example allocates device buffers with the CUDA Driver API,
wraps their addresses in CuTe pointers, and frees them before returning.
It compiles the launcher once, then calls the resulting function twice:

.. literalinclude:: cutlass_debugger_walkthrough.py
   :language: python
   :start-after: docs: start cutlass-debug-launches
   :end-before: docs: end cutlass-debug-launches

Each iteration resets the device output to ``-1``, launches the copy,
synchronizes, and checks all 256 values against NumPy. Here,
``cute.compile`` makes compilation an explicit step before either launch.
The two calls to ``compiled`` reuse that kernel.

A direct call to a CuTe ``@cute.jit`` function can
trace again to compute the module's cache key, even when compiled code
is reusable. Retaining the compiled callable makes the two GPU launches
independent of that tracing path.

Configure VS Code
^^^^^^^^^^^^^^^^^

Open this CCCL checkout as the VS Code folder. Use **Python: Select
Interpreter** to select an environment satisfying the
:ref:`CUTLASS compiler requirements <coop-cutlass-compiler-requirements>`
and the :doc:`programming guide's installation requirements <../coop_cutlass>`.
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
         "name": "cuda.coop CUTLASS: Debug active Python file",
         "type": "debugpy",
         "request": "launch",
         "program": "${file}",
         "console": "integratedTerminal",
         "cwd": "${workspaceFolder}",
         "justMyCode": false,
         "env": {
           "PYTHONPATH": "${workspaceFolder}/python/cuda_coop",
           "CUDA_COOP_CCCL_ROOT": "${workspaceFolder}",
           "CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION": "0",
           "CUDA_COOP_SOURCE_DUMP_DIR": "${workspaceFolder}/build/coop-cutlass-debug-sources"
         }
       }
     ]
   }

``justMyCode: false`` permits stepping into library code. ``PYTHONPATH``
and ``CUDA_COOP_CCCL_ROOT`` select this checkout's Python sources and C++
headers. ``cute.compile`` explicitly requests CuTe compilation. When run
as a script, the example also selects a fresh temporary provider-cache
directory and removes it on exit. A new process with that empty provider
cache lets each debug session reach NVRTC.
The Numba setting ``CUDA_COOP_ENABLE_CACHE=0`` does not control this
backend's provider cache.

On a machine with several GPUs, add ``CUDA_VISIBLE_DEVICES`` to ``env``.
Use the toolkit configured for your compatible compiler environment.

Set the first breakpoint on ``from cuda import coop`` in the example.
Keep that file active, select the named configuration in **Run and Debug**,
and press **F5**. ``${file}`` selects the active editor file when the
session starts. **F10** steps over, **F11** steps into, **Shift+F11** steps
out, and **F5** continues. Debug Console expressions use the selected
Call Stack frame. See the
`VS Code Python debugging documentation
<https://code.visualstudio.com/docs/python/debugging>`_ for those controls.

The stops below identify executable statements by function name, so they
remain useful as line numbers change. Set each breakpoint before continuing
past the stage that calls it. Avoid calling primitives from Debug Console:
they can emit additional IR into the paused trace.

Initial import and registration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The first stop is just before ``cuda.coop`` imports; the example has
already imported CuTe. Paths in this subsection are relative to
``python/cuda_coop/cuda/coop/``.

In ``__init__.py``, break on ``_auto_register_known_dsls()``. Continue and
step into the call. In ``_core/_auto_registration.py``, follow the loop
to ``candidate.activate()``. The CUTLASS candidate recognizes the already
loaded ``cutlass`` module and imports the qualified backend.

Before continuing, set a breakpoint on
``runtime = validate_cutlass_runtime()`` in
``cutlass/_compiler/_activation.py:register_trace_context()``. Step over
the runtime check and the following assignments. Inspect ``dsl`` and
``environment``. The function registers a predicate that recognizes
CuTe's active environment; registration itself does not compile a kernel.

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

Follow dispatch and the launch dimensions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In ``_core/api/_dispatch.py:_backend_module_name()``, break on
``owns_environment = probe()``. Continue from ``cute.compile``, then step over
the call. For ``module_name == "cuda.coop.cutlass"``,
``owns_environment`` is true. Inspect the Call Stack: CuTe is tracing
the kernel, and a common API call is selecting its backend. Disable this
breakpoint after seeing the selection; payload and group constructors
also pass through dispatch.

Paths for the remaining compiler stops are relative to
``python/cuda_coop/cuda/coop/cutlass/`` unless stated otherwise.
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
Step into ``current_kernel_launch_facts()`` on a fresh pass if you want
to follow that dependency boundary. These dimensions come from
``.launch(grid=1, block=128)`` and determine the CUB specialization.

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
that will implement it. Both runtime-control lists are empty: this copy
has no valid-prefix count, offset, or fill argument.

Step over request registration. It records the request in a session owned
by this trace and arranges for a finalization hook. No C++ has been
compiled yet.

Emit the device calls
^^^^^^^^^^^^^^^^^^^^^

Still in ``provider_load()``, stop on ``ffi(...)`` after
``scratch_types, scratch_args = _scratch_arguments(...)``. Both scratch
lists are empty. The FFI signature has two pointers: the source address
and a per-thread result buffer containing two integers.

Step over the complete FFI call and the following payload assignments.
CuTe has emitted a call into its IR, and ``output`` now holds the CuTe
values read from that result buffer. You are inspecting compiler values;
the GPU has not loaded a particular thread's integers yet.

Continue to the plan breakpoint for Store. It selects
``::cub::BLOCK_STORE_DIRECT``. In ``provider_store()``, break on its
``ffi(...)`` call and inspect ``values`` and ``request.symbol_name``.
Store passes the destination pointer and the two item values. Unlike
Load, it needs no result buffer. Disable the plan and emission breakpoints
after inspecting both operations.

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
buffer; Store builds its array from the two scalar arguments and calls
``BlockStore::Store``.

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

Set a breakpoint at the end of ``_trace_finalize_hook()`` on
``_bundle.append_link_library_attr(module, path)``. Continue and inspect
``path``. It names the cached ``.ltoir`` file containing both providers.
Step over the call and evaluate:

.. code-block:: python

   [(op.name, str(op.attributes["link-libraries"]))
    for op in module.body.operations if op.name == "gpu.module"]

The GPU module now names that file as a link input. CuTe's compiler and
final linker consume it to resolve the device calls. The example keeps
its temporary cache alive through compilation and both launches.
This handoff is a useful boundary when diagnosing a failure: reaching
it proves provider compilation finished, while successful kernel linking
and execution still lie ahead.

Finish the launch and observe reuse
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Disable the compiler breakpoints and set one on
``compiled(source_pointer, destination_pointer)`` in the example.
Continue. ``cute.compile`` has returned a callable; the first launch is
about to execute it with ``iteration == 0``. Continue again to reach
the second launch with ``iteration == 1``. The first verification message
has printed, and ``destination`` has been reset to ``-1`` on both host
and device.

Re-enable the plan and NVRTC breakpoints, then continue. Calling
``compiled`` executes the existing kernel without tracing or provider
compilation. Neither breakpoint should fire, and the second verification
message should print. Calling ``cute.compile`` again would request another
CuTe compilation, though the provider cache could supply its LTO-IR.
Start a new debug session to repeat the tour;
editing a file while paused does not replace the function already loaded
in the current process.

A second pass: shared scratch and synchronization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Add ``"args": ["--algorithm", "transpose"]`` to the launch configuration
and start a new debug session. From a terminal, the equivalent is:

.. code-block:: bash

   python docs/python/coop/cutlass_debugger_walkthrough.py --algorithm transpose

The kernel, launch dimensions, and expected output are unchanged. At the
plan stop, ``ALGORITHM`` now selects ``BLOCK_LOAD_TRANSPOSE`` or
``BLOCK_STORE_TRANSPOSE``. Storage ownership is ``CALLER``, and the
storage-reuse barrier is ``BLOCK``. The FFI signatures gain three scratch
arguments: a shared-memory address, its capacity in bytes, and the
automatic-synchronization flag.

Set these additional breakpoints before reaching Load:

* In ``_compiler/_storage.py:register_deferred_temp_storage_event()``,
  break on ``session.add_deferred_temp_storage_event(...)``. Inspect
  ``primitive_name``, ``requirement_key``, ``smem_addr_placeholder``, and
  ``size_placeholder``. The address and size are fresh IR placeholders;
  NVRTC has not yet supplied the C++ scratch layout. Load and Store record
  separate uses of the same ``TempStorage`` descriptor.
* In ``_compiler/_finalize.py:_trace_finalize_hook()``, break on
  ``plans = _storage.plan_deferred_temp_storage_events(...)``. Inspect
  ``compilation.layouts``. NVRTC has compiled both providers and recovered
  their exact C++ ``sizeof`` and ``alignof`` requirements from the same
  program. Step over planning and inspect ``plans``: there is one shared
  allocation, sized for the larger requirement. Its two bindings have
  byte offset zero. The size and alignment come from this build; do not
  copy fixed numbers from another toolkit.
* In ``_compiler/_storage.py:materialize_deferred_temp_storage_plans()``,
  break on ``smem_ptr = allocator.allocate(...)``. Inspect
  ``plan.size_in_bytes``, ``plan.alignment``, and ``plan.bindings``. Step
  through allocation and operand replacement to see the recorded calls
  acquire their actual address and capacity.

At the link-input stop, compare ``print(module)`` with the Direct version.
There is now a shared allocation and the calls carry scratch operands.
Open the dumped C++ and find ``temp_storage_auto_sync``. Each wrapper
ends with a conditional ``__syncthreads()``; this example passes a true
flag, so Load finishes using the workspace before Store reuses it.

CUTLASS emits the trailing barrier in
the provider wrapper and patches deferred storage operands after obtaining
the C++ layouts. ``ThreadData`` still holds the per-thread payload;
``TempStorage`` describes the shared workspace used during each primitive.
Continue to verify both launches of the transpose copy.

If a breakpoint does not stop
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Check ``coop.__file__`` at the first launch. The executing module and
  breakpoint must belong to the same checkout. Inspect ``cutlass.__file__``
  too if a required compiler interface is missing.
* Use the named launch configuration with ``justMyCode: false``. A hollow
  breakpoint can remain pending until its module loads. Put it on an
  executable statement rather than a decorator or ``def`` line.
* If registration is skipped, keep the example's import order and
  ``CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION=0``. Read any dependency error
  or registration warning before investigating later compiler stops.
* If NVRTC is skipped, restart the complete script. Its temporary provider
  cache is created only under ``if __name__ == "__main__"``. Check that
  the provider directory is new and that you set the breakpoint before
  ``cute.compile``. A stop in ``_compile_bundle_source()`` can show a
  provider-cache hit. A source dump alone does not prove NVRTC ran: the
  backend writes the dump before looking up a cached provider.
* If a later launch retraces, check that you are calling ``compiled``.
  This walkthrough retains the callable returned by ``cute.compile``
  and invokes it twice; calling the decorated ``launch`` directly follows
  CuTe's JIT path instead.

Working on a primitive
----------------------

Start with the C++ overload and a small kernel exercising it. Check the
shared family's plan for payload layout, participation, result ownership,
and storage requirements. Add a shared declaration under ``_core/api``
only when the common API needs it; keep CUTLASS-specific controls in the
qualified entry point and its ``.pyi`` file.

Adapt the shared plan in ``cutlass/_lowering``. The lowering must agree
with its provider wrapper about argument types, result buffers, and
scratch operands. Register the typed request with the current session
and preserve rollback if emission fails. Load/Store illustrates in-place
payloads; Scan and Reduce show result contracts and additional controls.
Reuse the existing renderer registration, bundle finalizer, and storage
planner when their contracts express the operation.

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
