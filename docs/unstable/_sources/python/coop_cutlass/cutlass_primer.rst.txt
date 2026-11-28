.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _cuda.coop.cutlass.cutlass_primer:

Introduction to CUTLASS for cuda.coop.cutlass Developers
========================================================

This page is for developers who work on ``cuda.coop.cutlass`` and have not
used CUTLASS or its Python DSL before. It explains what CUTLASS is, how the
CuTe DSL turns a Python function into a GPU kernel, how that differs from
other Python compilers you may know, and which of its mechanisms
``cuda.coop.cutlass`` relies on. The :doc:`Developer Guide <developer_guide>`
then follows one primitive through the implementation, and
:doc:`cutlass_integration_example` shows, in a single file with no
``cuda.coop.cutlass`` abstractions, the first eight integration points of
the table in "Where cuda.coop.cutlass plugs in".

Paths to CUTLASS DSL modules below are relative to the installed ``cutlass``
package, for example ``site-packages/nvidia_cutlass_dsl/dsl_packages/cutlass``
or ``python_packages/cutlass`` depending on the wheel. Paths to our modules
are relative to ``python/cuda_coop_cutlass/cuda/coop/cutlass``. The DSL is
distributed under NVIDIA's EULA, so read its sources for understanding but
do not copy them into this repository.

What CUTLASS is, and which part we use
--------------------------------------

CUTLASS is NVIDIA's open-source C++ template library for GEMM and related
kernels. Its lower layer, **CuTe**, is an algebra of layouts and tensors that
describes how threads and data map onto each other. ``cuda.coop.cutlass``
does not use the C++ library. CUB, the library that ``cuda.coop.cutlass``
wraps, is a separate CCCL library.

Since CUTLASS 4, the same project ships a Python front end, the **CUTLASS
DSL** or **CuTe DSL**, installed as ``nvidia-cutlass-dsl``. You write GPU
kernels as decorated Python functions, and the DSL compiles them through its
own MLIR distribution to a cubin that it also launches. This Python layer is
the only part of CUTLASS that ``cuda.coop.cutlass`` depends on. When this
documentation says CuTe, CuTe DSL, or the CUTLASS compiler, it means this
Python-plus-MLIR toolchain.

Kernel code imports one package, ``cutlass``. The parts that matter here
are:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Part
     - Role
   * - ``cutlass`` and ``cutlass.cute``
     - Public API: ``cute.jit``, ``cute.kernel``, ``cute.compile``, numeric
       types such as ``cutlass.Int32``, ``cutlass.Constexpr``, pointers,
       layouts, tensors, ``cute.arch`` intrinsics, and ``cute.ffi``.
   * - ``cutlass.base_dsl``
     - The generic tracing compiler: the ``@jit`` and ``@kernel`` runners,
       the AST preprocessor, compile options, the FFI, and the environment
       variable manager.
   * - ``cutlass.cutlass_dsl``
     - The CuTe-specific DSL subclass ``CuTeDSL``: GPU module and kernel
       emission, kernel launch, launch facts, and the compiler backend.
   * - ``cutlass._mlir``
     - MLIR Python bindings with the ``cute``, ``cuda``, ``lir``, ``nvvm``,
       ``llvm``, ``scf``, ``arith``, ``func``, and ``gpu`` dialects.
   * - ``cutlass.memory``
     - ``SmemAllocator``, the shared-memory allocation helper.
   * - ``_cutlass_ir`` and ``libcute_dsl_runtime``
     - The native code: the extension module behind ``cutlass._mlir`` and
       the runtime library. The ``[cu13]`` extra installs their CUDA 13
       builds.

``cuda.coop.cutlass`` adds one thing to this toolchain: CUB device code
reachable from a CuTe kernel, with the kernel's launch shape, scratch, and
synchronization requirements handled for the caller.

A CuTe kernel, piece by piece
-----------------------------

The kernel from :doc:`cutlass_integration_example` is small enough to read
whole:

.. literalinclude:: cutlass_integration_example.py
   :language: python
   :start-after: docs: start cute-cub-kernel
   :end-before: docs: end cute-cub-kernel

``@cute.kernel`` marks a GPU kernel and ``@cute.jit`` marks a host function
that CuTe compiles. Calling the kernel inside the host function returns a
launcher. ``.launch(grid=..., block=...)`` emits the kernel into the module
and emits the launch. ``cute.compile(launch, source, destination)`` returns
a compiled callable for those argument types, which is then called with
matching arguments. Calling ``launch(...)`` directly compiles and runs in
one step.

Inside the kernel, ``cute.arch.thread_idx()`` returns three ``cutlass.Int32``
values, ``cute.make_tensor`` wraps a pointer in a layout, and indexing a
tensor loads or stores one element. ``block_sum`` is the function this
example adds. It is an ordinary Python function and the DSL has no special
knowledge of it. The table below says what each object is while the function
is being compiled.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - In the Python source
     - During tracing
     - In the MLIR module
   * - ``cutlass.Int32``, ``cutlass.Float32``, ...
     - A Python object holding a constant or an ``ir.Value`` of the matching
       MLIR type. Arithmetic on symbolic values emits ``arith`` operations.
     - ``i32``, ``f32``, ... SSA values.
   * - ``cute.Pointer``
     - A typed pointer with an address space and alignment.
     - ``!cute.ptr<i32, gmem, align<16>>``.
   * - ``cute.Tensor`` from ``make_tensor``
     - A pointer plus a static or dynamic layout.
     - ``!cute.memref<...>`` views and ``cute.memref.load`` or ``store``.
   * - Python literals and ``Constexpr`` values used inside the body
     - Ordinary Python values, folded at trace time.
     - Constants or nothing at all.
   * - ``cutlass.Constexpr`` parameter
     - A Python value that specializes the function.
     - No MLIR argument. The value is part of the function's cache identity.
   * - ``grid=1, block=(32, 4, 1)``
     - Python integers in a ``LaunchConfig``.
     - The ``cutlass_launch_facts`` and ``nvvm.reqntid`` kernel attributes
       and the ``cuda.launch_cfg.create`` operands.

Ordinary function arguments can become dynamic DSL values: an unannotated
Python ``int`` argument is adapted to ``Int32``, and tuple elements are
adapted recursively. Use ``Constexpr`` for parameters that must specialize
the function at trace time.

How the DSL compiles a function
-------------------------------

The DSL compiles by **tracing**: it runs your Python function with symbolic
arguments and records the operations that run. The run happens in three
steps that matter to us.

Step 1: a source-level rewrite of control flow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

On the first call of a decorated function, ``BaseDSL.run_preprocessor``
fetches its source with ``inspect``, parses it into a Python AST, and runs
``DSLPreprocessor``, an ``ast.NodeTransformer`` in
``base_dsl/ast_preprocessor.py``. The transformer rewrites control flow:

* ``for i in range(...)`` and ``cutlass.range(...)`` become a staged loop
  that emits ``scf.for``, even with Python integer bounds. Variables assigned
  in the body become loop-carried values. ``cutlass.range_constexpr(...)``
  stays a Python loop, so the body is traced once per iteration and the
  result is unrolled code. A ``for`` over a list or tuple also stays a Python
  loop. An ``if``, ``while``, or ``for`` inside it that holds a ``break``,
  ``return``, or ``raise`` stays in Python too.
* ``if`` and ``while`` become ``scf.if`` and ``scf.while`` with ``scf.yield``
  of every variable the branches assign, even when the condition is a Python
  value. A condition written as ``cutlass.const_expr(x)`` asserts that ``x``
  is a Python value, so the branch is decided at trace time and only one side
  is traced.
* ``and``, ``or``, and conditional expressions on DSL values are rewritten
  so that Python's short-circuiting does not silently drop the dynamic
  operand.
* ``return``, ``break``, ``continue``, and ``raise`` inside staged control
  flow are rejected with a diagnostic by this preprocessing mode.

The rewritten AST is compiled and swapped into the function object's
``__code__``. The rewrite happens once per function and preprocessing mode.
Later calls reuse the rewritten code. So the DSL does analyze the AST,
primarily to stage control flow. It also rewrites selected Python calls and
expressions. Step 2 type-checks the function and decides which operations
are legal.

Step 2: execution with symbolic values
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``BaseDSL.generate_original_ir`` creates an ``ir.Module``, a ``gpu.module``
for kernels, and a host ``func.func`` whose block arguments stand for the
non-``Constexpr`` parameters. It wraps those arguments in the DSL's Python
types and then calls the rewritten Python function with them. Every
operation on a DSL value emits MLIR at ``ir.InsertionPoint.current``. Every
Python-only value behaves as it would in plain Python. When the function
returns, the module contains the full program.

Two consequences matter to ``cuda.coop.cutlass``:

* **Any Python function called in the body runs at trace time.** A call to
  ``coop.load`` executes as ordinary Python. It inspects its arguments and
  emits IR itself. There is no typing registry or lowering table to register
  with, as there is in Numba.
* **The kernel body is traced inside the launch.** ``KernelLauncher.launch``
  in ``cutlass_dsl/cutlass.py`` receives the grid and block, attaches them
  to the kernel operation as the ``cutlass_launch_facts`` attribute, and only
  then traces the kernel body. Code running in the body can therefore read
  the exact launch shape through ``cute._get_launch_facts()``. When any
  component of the block, grid, or cluster shape is a DSL value, the facts
  omit that whole shape.

The example kernel's ``if rank == 0`` illustrates the two steps together.
Step 1 rewrote the statement. In step 2, the comparison on the DSL value
``rank`` emitted ``arith.cmpi`` and the rewritten ``if`` emitted ``scf.if``:

.. code-block:: text

   %9 = arith.cmpi eq, %6, %c0_i32_9 : i32
   %10 = scf.if %9 -> (!memref_gmem_i32_1) {
     cute.memref.store(%view_6, %coord_13, %8) : (...) -> ()
     scf.yield %view_6 : !memref_gmem_i32_1
   } else {
     scf.yield %view_6 : !memref_gmem_i32_1
   }

Step 3: finalize, hash, compile, link, run
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After the body returns, ``generate_original_ir`` runs the registered
**trace-finalize hooks** with ``hook(dsl, module, function_name)``. A hook
may inspect or change the finished module. Only then does the DSL compute
the **module hash** that keys its JIT cache. If a cached compilation exists
for that hash, it is reused. Otherwise the module goes through the compile
pipeline (``lir-to-cute-dsl``, ``cute-to-nvvm``, and the LLVM backend),
device code is linked with nvJitLink, and the result is loaded as a
``JitCompiledFunction`` and executed. When ``generate_original_ir`` returns,
before any cache lookup, the DSL merges each ``gpu.module``'s
``link-libraries`` attribute into its ``LinkLibraries`` compile option, which
is how externally compiled device code reaches the device link.

Because the cache key is the traced module, **a direct call always traces
again**, even on a cache hit. Tracing is how the DSL learns whether the
program changed. ``cute.compile`` traces and compiles on every call and
skips the hash and the JIT cache. It returns the compiled callable, so
calling that object launches without tracing. Our tests and examples use
``cute.compile`` when they need to separate compilation from execution.

How this differs from Dynamo, Numba, and Triton
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A common first question is whether CuTe traces the way PyTorch Dynamo does.
It does not, and the differences matter when deciding where to hook in.

.. list-table::
   :header-rows: 1
   :widths: 16 28 28 28

   * - System
     - What it analyzes
     - How it captures a program
     - How it reuses a compilation
   * - PyTorch Dynamo
     - CPython bytecode, intercepted through the frame evaluation API before
       a frame runs.
     - Symbolically interprets bytecode into an FX graph. Unsupported code
       causes a graph break and falls back to eager Python.
     - Guards on argument properties. When they pass, the compiled graph
       runs without re-executing the Python.
   * - Numba CUDA (``numba-cuda``)
     - CPython bytecode, lifted to Numba IR and type-inferred.
     - Compiles the whole function per argument type signature. Unsupported
       operations are typing errors. Extensions register typing templates
       and lowering functions.
     - Dispatch on argument types to a previously compiled specialization.
   * - Triton
     - The function's AST.
     - A code generator visits the AST and emits Triton MLIR without running
       the body. ``tl.constexpr`` arguments specialize.
     - Cache keyed by signature, constexpr values, and options.
   * - CuTe DSL
     - The AST, primarily for control-flow staging. Then the running program.
     - Rewrites staged control flow, then executes the Python body with
       symbolic values. Unsupported operations raise during execution.
       Everything in the body is captured. There are no graph breaks.
     - Re-traces on every direct call and looks up the module hash. The
       callable returned by ``cute.compile`` launches without tracing.

For ``cuda.coop.cutlass`` the practical differences are these. There are no
guards, so a primitive can read any Python-level fact it wants at trace time
and the DSL will not try to generalize over it. Our planners use that to
specialize on the exact block shape and dtype. There are no graph breaks, so
a primitive cannot defer to the interpreter. It must emit valid IR or
raise. Numba extensions register typing templates and lowering functions.
Our CuTe integration registers neither. ``coop.load`` is a function that is
called during tracing.

PyIR, an optional second preprocessing mode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CuTe DSL 4.9 has **PyIR**, off by default. It is enabled with
``CUTE_DSL_ENABLE_PYIR=1``, inside a ``with CuTeDSL.enable_pyir():`` block,
or for one compilation with ``cute.compile[cute.FrontendNext](...)``.
``PyIRDSLPreprocessor`` extends the AST rewrite to put the function into SSA
form, the trace emits ``pyir`` dialect operations, and the compile pipeline
lowers them with ``pyir-prelower`` and ``convert-pyir-to-scf``. The
preprocessing mode is part of the cached rewrite's identity, so switching
modes re-runs the rewrite.

The primitive entry points stage no control flow of their own. Other parts
of the package still depend on the preprocessing mode. In particular, the
``TempStorage.release()`` lifetime analysis runs at trace finalization,
before PyIR is lowered. It runs whenever a ``TempStorage`` is constructed
during the trace, and each call with compiler-managed deferred scratch
constructs one. It recognizes ``scf`` and ``cf`` operations and rejects any
other operation that has regions or successors. The integration described
here uses the default preprocessing mode. PyIR needs separate qualification,
including storage lifetimes across control flow.

Where cuda.coop.cutlass plugs in
--------------------------------

This section maps each need of the integration to the DSL facility that
serves it and the module that uses it. :doc:`cutlass_integration_example`
exercises the first eight rows in one file.

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * - Need
     - CuTe DSL facility
     - Where we use it
   * - Run code when the user calls a primitive
     - Trace-time execution; ``ir.InsertionPoint.current`` names the block
       being built
     - ``_group_*.py`` entry points, ``_compiler/_state.py``
   * - Know the launch shape while planning
     - ``cute._get_launch_facts()`` reading the ``cutlass_launch_facts``
       kernel attribute
     - ``_compiler/_launch.py``, ``_planning/launch.py``
   * - Call C++ that is not compiled yet
     - ``cute.ffi``: a private ``func.func`` prototype in the ``gpu.module``
       and a ``func.call``
     - ``_lowering/*.py``
   * - Pass values across the boundary
     - DSL numerics wrap ``ir.Value``; ``cute.Pointer`` converts to
       ``!llvm.ptr``; ``cute.make_rmem_tensor`` gives an addressable
       per-thread buffer
     - ``_compiler/_types.py``, ``_thread_data.py``
   * - Run after tracing, before hashing
     - ``CuTeDSL.register_trace_finalize_hook``
     - ``_compiler/_state.py``, ``_compiler/_finalize.py``
   * - Supply the C++ definition to the device link
     - ``gpu.module`` ``link-libraries`` attribute, merged into the
       ``LinkLibraries`` compile option
     - ``_compiler/_bundle.py``, ``_compiler/_finalize.py``
   * - Allocate shared memory for CUB
     - ``SmemAllocator.allocate`` at the kernel entry block and
       ``ir.Value.replace_all_uses_with``
     - ``_compiler/_storage.py``, ``_compiler/_layout.py``
   * - Select the target architecture
     - The ``GPUArch`` compile option or ``dsl.envar.arch``
     - ``_compiler/_target.py``
   * - Keep a wrapper call convergent
     - The ``llvm`` dialect: ``llvm.func`` and ``llvm.call`` with the
       ``convergent`` attribute
     - ``_compiler/_convergence.py``
   * - Raise an explicit launch size to cover deferred scratch
     - The shared-memory operand of ``cuda.launch`` and
       ``cuda.launch_cfg.create``, and ``cute.kernel_smem_size``
     - ``_compiler/_storage.py``
   * - Compile a user's operator to LTO-IR
     - ``cute.compile_to`` with ``TargetKind.Device``, and
       ``CuteCompiler``
     - ``_device_operator.py``

Entry points are ordinary functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``coop.load``, ``coop.reduce``, and the other public functions validate
their Python arguments, plan the CUB specialization, and emit IR, all while
the user's kernel is being traced. Nothing is registered with the DSL before
the first call. To find the trace they belong to, they walk from
``ir.InsertionPoint.current.block.owner`` through parent operations to the
enclosing ``builtin.module``. To find the kernel that owns their scratch,
they stop at the kernel operation, which in CuTe DSL 4.9 is an
``lir.func`` carrying the ``gpu.kernel`` attribute, or a ``cuda.kernel``
when ``CUTE_DSL_USE_EXTENSION_COMPILER=0``. ``BundleSession`` in
``_compiler/_state.py`` keeps per-trace state keyed by the DSL's compile
options object and that module, so a second trace on the same options
object gets its own session. A nested ``@cute.jit`` call is traced into
its caller's module, so it shares the caller's session and bundle. Several
kernels in one module share them too. Each scratch use records its kernel
operation, so each kernel gets its own allocations.

Launch facts
^^^^^^^^^^^^

CUB block primitives are templates on the block shape, and warp primitives
need to know how many complete warps a block holds. The planner therefore
requires the exact dimensions, which ``_compiler/_launch.py`` reads through
``cute._get_launch_facts()`` and records with their provenance. The
``nvvm.reqntid`` attribute also specifies an exact block shape, whereas
``nvvm.maxntid`` supplies only an upper bound. The planner takes launch
facts only from the launch facts API. An exact shape fixes how many
threads the block has. The kernel must still bring every thread of the
group to each primitive call. See :ref:`coop-cutlass-exact-launch-facts`.

Calls through the FFI
^^^^^^^^^^^^^^^^^^^^^

``cute.ffi(name=..., params_types=[...], return_type=...)`` returns a callable
that, when invoked with DSL values, declares a private function prototype in
the enclosing ``gpu.module`` if needed and emits ``func.call``. Parameter and
return types are DSL numeric types or MLIR types. A ``cute.Pointer`` argument
is converted to ``!llvm.ptr`` automatically, and a pointer in a specific
address space is cast to the generic address space when the prototype
declares one. The symbol is used verbatim, which is
why generated wrappers are ``extern "C"`` and why request objects derive
deterministic symbol names from their plan.

Load returns several values per thread. The lowering allocates a register
tensor with ``cute.make_rmem_tensor``, passes its ``.iterator.llvm_ptr`` as
a result-buffer argument, and after the call reads the elements back into
the ``ThreadData`` payload. The DSL also offers ``cute.extern``, a decorator
that mangles symbol names from argument types. We call ``cute.ffi``
because our symbol names must also identify the CUB plan.

The trace-finalize hook
^^^^^^^^^^^^^^^^^^^^^^^

``register_trace_finalize_hook`` is persistent for the DSL instance and
ignores duplicate registrations. ``cuda.coop.cutlass`` registers one
dispatcher and points it at the current finalizer, so repeated imports or
activations never accumulate callbacks. The hook runs for every trace. It
first removes, from the DSL's ``LinkLibraries`` option, the paths the
package attached in earlier traces. It then takes the session for the
module it was given and returns if there is none or the session is empty.
For a session with requests, it checks ``TempStorage`` lifetimes, renders
all wrappers into one C++ translation unit, compiles them with NVRTC to
LTO-IR, and materializes deferred scratch. ``_compiler/_convergence.py``
then replaces each wrapper's ``func.func`` declaration and ``func.call``
operations with ``llvm.func`` and ``llvm.call`` operations marked
``convergent``. Last, the hook attaches the artifact path to each
``gpu.module``. The LTO-IR file of each ``DeviceOperator`` the trace uses
is attached the same way.

The DSL's own module hash runs after the hook and includes the module's
attributes, so a different bundle path changes the hash. The key of our
bundle cache is built in ``_compiler/_bundle.py`` from the rendered source,
the header and toolkit identity, the ``sizeof`` and ``alignof`` probes that
query scratch layouts, and the compiler options, which include the
architecture. ``_compiler/_cache.py`` stores and checks the file. A repeated
kernel finds the same ``.ltoir`` file, so on a direct call the DSL's cache
hits as well. ``cute.compile`` turns the DSL's cache off.

LTO-IR and the device link
^^^^^^^^^^^^^^^^^^^^^^^^^^

NVRTC compiles the wrappers with ``--relocatable-device-code=true`` and
``-dlto`` and we store the LTO-IR container in a cache directory. The DSL
documents its ``link-libraries`` mechanism in terms of LLVM bitcode, but its
compiler recognizes LTO-IR containers as nvJitLink inputs and links them
with the kernel's own LTO-IR. The link can inline the wrapper into the
kernel. The runtime tests run ``cuobjdump`` on the final cubin and fail if
a provider symbol remains.

CuTe DSL 4.9 copies compile options for each invocation, so
adding a module's link libraries does not mutate the reusable option
template. The ``DeviceTarget`` compile mode, which compiles a
``@cute.jit`` function as a device function for export, rejects
``link-libraries``. Kernels that call ``cuda.coop.cutlass`` primitives use
the default host-plus-kernel compilation. ``compile_operator`` compiles its
function as a device function, through ``cute.compile_to`` with
``TargetKind.Device``. That function cannot call a primitive.

Shared memory
^^^^^^^^^^^^^

A CUB primitive's ``TempStorage`` has a size and alignment that only the C++
compiler knows. Two routes exist. A wrapper can declare its own
``__shared__`` storage in C++, which the device link lays out alongside the
kernel's shared memory. Warp Scan and the other wrappers that own their
storage use this route. The deferred route serves every call whose scratch
comes from a ``TempStorage`` descriptor, whether the caller supplies it or
the lowering constructs one. Tracing emits placeholder ``arith.constant``
operands for the address and size. The NVRTC program that compiles the
wrappers reports the exact layout through ``nvrtcAddNameExpression`` and
``nvrtcGetLoweredName``. Finalization then inserts a
``SmemAllocator().allocate(size, alignment)`` at the kernel's entry block
and replaces the placeholders. ``SmemAllocator`` is stateless: it emits
``cute.memref.alloca`` operations that the compiler lays out together with
the user's allocations, and the launch reads the total through
``cute.kernel_smem_size``. A launch that passes an explicit ``smem=`` size
was traced before these allocations existed, so finalization raises that
operand to at least that total. See :ref:`coop-cutlass-scratch-allocation`.

Barriers are emitted inside the wrappers, as ``__syncthreads()`` or a masked
``__syncwarp()``, so the kernel's IR contains no barrier operations for
them. Group queries and group synchronization lower to small wrappers over
``cuda::hierarchy`` and the CUDA barrier intrinsics.

What we deliberately avoid
^^^^^^^^^^^^^^^^^^^^^^^^^^

``cuda.coop.cutlass`` defines no MLIR dialect, adds no passes to the compile
pipeline, and does not generate CuTe IR for the CUB algorithms themselves.
The integration surface is the trace-time Python call, the launch facts,
the FFI, the finalize hook, the link attribute, the shared-memory
allocator, and the target architecture setting. Keeping to those means a
CUTLASS release can change its internal lowering without affecting the
generated CUB code. Inside the hook the
package also rewrites two things CuTe traced. Wrapper calls become
``llvm.call`` operations marked ``convergent``. Each launch of a kernel with
deferred scratch requests at least the size that ``cute.kernel_smem_size``
reports. The launch rewrite depends on the operand positions of CuTe's
``cuda.launch`` and ``cuda.launch_cfg.create`` operations, so recheck it
when the CUTLASS dependency changes. ``compile_operator`` adds
``cute.compile_to`` to the integration surface.

Private interfaces we depend on
-------------------------------

Several of these facilities are private or new in CuTe DSL 4.9. Their names
begin with an underscore or appear only in its sources. The installed
package is tested against CuTe DSL 4.9.0.dev1.
``_compiler/_runtime.py`` checks at import that the functions and classes in
the first four items of the list below exist, and reports which are missing.
Each time ``_compiler/_storage.py`` allocates deferred scratch, it checks
the fifth item's bindings except ``ir.Operation.walk``, ``llvm.LLVMFuncOp``,
and ``llvm.CallOp``. No capability check covers those three:

* ``cutlass.cutlass_dsl.CuTeDSL._get_dsl``,
  ``register_trace_finalize_hook``, and ``trace_finalize_hooks``.
* ``cutlass.cute._get_launch_facts`` and its ``cutlass_launch_facts``
  schema in ``cutlass_dsl/_launch_facts_metadata.py``.
* ``cutlass.base_dsl.compiler.LinkLibraries`` with the ``link-libraries``
  option name, and ``GPUArch``.
* ``cutlass.base_dsl.common.get_current_env_manager``, which the tests call
  to check that a trace released its environment.
* The MLIR bindings: ``ir.InsertionPoint.at_block_begin``,
  ``ir.Value.replace_all_uses_with``, ``ir.Operation.walk``, the ``arith``
  and ``llvm`` dialects, ``llvm.LLVMFuncOp`` and ``llvm.CallOp`` with their
  ``convergent`` keyword, ``cute.Pointer.to_llvm_ptr``, and
  ``SmemAllocator.allocate``.
* The launch operations the hook rewrites: ``cuda.launch``,
  ``cuda.launch_ex`` with ``cuda.launch_cfg.create``, the position of their
  dynamic shared-memory operand, and ``cute.kernel_smem_size``. Nothing
  checks these at import.
  ``python/cuda_coop_cutlass/tests/compile/test_storage.py`` reads the
  shared-memory size from the lowered launch.
* For ``compile_operator``: ``cute.compile_to`` and, in
  ``cutlass.compiler``, ``CuteCompiler``, ``ArtifactType``, ``TargetKind``,
  ``Var``, and ``DType``. ``compile_operator`` checks for them when it is
  called.

When a CUTLASS release changes one of these, expect the failure at import or
at the first compilation, before any kernel runs. Start with
``validate_cutlass_runtime`` and the lifecycle tests named in
:doc:`developer_guide`. A renamed ``cuda.launch`` or ``cuda.launch_ex`` is
the exception. The hook then finds no launch to rewrite, an explicit
``smem=`` size stays as it was, and ``tests/compile/test_storage.py`` is the
only check that fails. The :ref:`FAQ <coop-faq-compiler-setup>` lists the
other reasons a kernel can fail to compile after a successful import.

Reading the CUTLASS DSL sources
-------------------------------

Read these modules to see how a trace behaves:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Module
     - What to look for
   * - ``base_dsl/dsl.py``
     - ``BaseDSL.jit_runner`` and ``_preprocess_and_replace_code`` for the
       decorator path; ``generate_original_ir`` for tracing, hooks, and
       hashing; ``_merge_gpu_link_libraries``;
       ``register_trace_finalize_hook``.
   * - ``base_dsl/ast_preprocessor.py`` and ``ast_helpers.py``
     - The control-flow rewrite, ``range_constexpr``, ``const_expr``, and
       the early-exit diagnostics.
   * - ``cutlass_dsl/cutlass.py``
     - ``CuTeDSL``, ``KernelLauncher.launch``, ``_build_launch_facts_attr``,
       and the ``nvvm.reqntid`` derivation.
   * - ``cute/launch_facts.py``
     - ``_get_launch_facts`` and the kernel operation names it accepts.
   * - ``base_dsl/ffi.py`` and ``cute/ffi.py``
     - ``FFI.__call__``: prototype insertion, type checks, and the pointer
       conversions.
   * - ``base_dsl/compiler.py``
     - ``CompileOptions``, ``LinkLibraries``, ``GPUArch``, ``DeviceTarget``,
       and the ``cute.compile[...]`` option syntax.
   * - ``memory/smem.py``
     - ``SmemAllocator`` and the ``smem.partition_id`` attribute.
   * - ``base_dsl/env_manager.py``
     - The ``CUTE_DSL_*`` environment variables of the base DSL and whether
       each affects the cache key. ``cutlass_dsl/cutlass_env_manager.py``
       adds the CuTe-specific ones.

Useful environment variables while debugging a trace, all prefixed
``CUTE_DSL_``: ``PRINT_IR=1`` prints each module after tracing; ``KEEP`` with
``ir``, ``ptx``, ``cubin``, or ``sass`` keeps those artifacts under
``DUMP_DIR``; ``ARCH`` overrides the detected target; ``LOG_TO_CONSOLE=1``
turns on the DSL's own logging; ``LOG_LEVEL`` sets its threshold.
``CUDA_COOP_CUTLASS_SOURCE_DUMP_DIR`` dumps the generated C++ on our side.
See :doc:`configuration`.

Vocabulary
----------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Term
     - Meaning in this documentation
   * - trace
     - One execution of a ``@cute.jit`` function with symbolic arguments,
       producing one MLIR module. A ``@cute.jit`` function it calls is
       traced into the same module. A kernel is traced inside its launch.
   * - module, ``gpu.module``, kernel operation
     - The ``builtin.module`` for a trace; the ``gpu.module`` holding its
       kernels; the ``lir.func`` or ``cuda.kernel`` operation for one kernel.
   * - staged control flow
     - An ``if`` or ``while``, or a ``for`` over ``range(...)`` or
       ``cutlass.range(...)``, rewritten by the preprocessor to emit ``scf``
       operations. ``const_expr`` conditions and ``range_constexpr`` loops
       stay in Python.
   * - provider
     - A generated ``extern "C"`` C++ device function that calls CUB. For a
       group query or barrier it calls ``cuda::hierarchy`` or a CUDA barrier
       intrinsic. Calls whose requests are equal share one provider.
   * - provider request
     - The record describing one provider: its plan, dtype, symbol name,
       and the C++ renderer that emits it.
   * - bundle, session
     - The C++ translation unit containing all providers of one trace, and
       the per-trace object that collects requests and scratch uses for it.
   * - trace-finalize hook
     - The DSL callback that runs after tracing and before module hashing.
       ``cuda.coop.cutlass`` compiles the bundle there.
   * - deferred scratch
     - Shared memory whose placeholder operands are traced first and replaced
       after NVRTC reports the C++ layout.
   * - link library
     - A file named in a ``gpu.module``'s ``link-libraries`` attribute that
       the device link consumes. Ours are the bundle's LTO-IR and the LTO-IR
       of each ``DeviceOperator`` the trace uses.
