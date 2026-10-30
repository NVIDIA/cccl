.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _cuda.coop.cutlass.integration_example:

Minimal CuTe DSL Integration Example
====================================

``cuda.coop.cutlass`` integrates with the CUTLASS CuTe DSL through a handful
of mechanisms that are easy to lose among planners, thread groups, payloads,
and caches. This page isolates them. One script,
:download:`cutlass_integration_example.py <cutlass_integration_example.py>`,
calls ``cub::BlockReduce`` from a CuTe kernel using only the DSL's own
interfaces, NVRTC, and the CCCL headers. It imports nothing from
``cuda.coop``.

The intended readers are CUTLASS developers reviewing whether this is how
the DSL is meant to be extended, and ``cuda.coop`` developers who want the
mechanisms without the abstractions. The closing section lists the
questions we would like CUTLASS reviewers to answer.

Run it from a CCCL checkout with ``nvidia-cutlass-dsl``, ``cuda-bindings``,
and NumPy installed:

.. code-block:: bash

   python docs/python/coop_cutlass/cutlass_integration_example.py
   # block sum verified: 3008

   CUTE_CUB_EXAMPLE_PRINT_IR=1 python docs/python/coop_cutlass/cutlass_integration_example.py

The second command prints the finalized MLIR module from inside the hook.
``CUDA_COOP_CCCL_ROOT`` selects another CCCL checkout for the headers.

What the script uses from the DSL
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 44 14 42

   * - Interface
     - Status
     - Purpose
   * - ``@cute.kernel``, ``@cute.jit``, ``.launch``, ``cute.compile``
     - public
     - Write, compile, and launch the kernel.
   * - ``cute.ffi``
     - public
     - Emit a call to a symbol defined later.
   * - ``cutlass.memory.SmemAllocator``
     - public
     - Allocate CUB scratch in shared memory.
   * - ``cute._get_launch_facts()``
     - private
     - Read the exact block shape while tracing the body.
   * - ``CuTeDSL._get_dsl().register_trace_finalize_hook``
     - private
     - Run after tracing, before the module hash and device link.
   * - ``gpu.module`` ``link-libraries`` attribute
     - internal
     - Hand the compiled LTO-IR to the device link.
   * - ``cutlass._mlir`` (``ir``, ``arith``, ``llvm``)
     - internal
     - Trace placeholders, insert the allocation, replace operands.
   * - ``dsl.envar.arch``
     - internal
     - Select the NVRTC target from the DSL's own setting.

The kernel
----------

The kernel sums one ``int`` per thread across a ``(32, 4, 1)`` block and
thread zero stores the total. ``block_sum`` is the integration; everything
else is ordinary CuTe.

.. literalinclude:: cutlass_integration_example.py
   :language: python
   :start-after: docs: start cute-cub-kernel
   :end-before: docs: end cute-cub-kernel

Steps 1 to 3: the trace-time entry point
----------------------------------------

The DSL compiles by executing the kernel body with symbolic values, so
``block_sum`` runs while CuTe traces ``sum_block``. It reads the launch
shape, registers the finalize hook, traces two placeholder constants for
the scratch, and emits the call:

.. literalinclude:: cutlass_integration_example.py
   :language: python
   :start-after: docs: start cute-cub-entry
   :end-before: docs: end cute-cub-entry

Three details matter here.

**The block shape is exact.** ``KernelLauncher.launch`` attaches a
``cutlass_launch_facts`` attribute to the kernel operation before it traces
the body, so ``_get_launch_facts()`` returns ``(32, 4, 1)`` as Python
integers. CUB's ``BlockReduce`` is a template on those dimensions, and a
mismatch between the template and the launch would be silent corruption.
The DSL also derives ``nvvm.reqntid`` from the same configuration.

**The call is a plain ``func.call``.** ``cute.ffi`` declares a private
``func.func`` prototype in the enclosing ``gpu.module`` on first use and
emits the call. The symbol is used verbatim, so the C++ definition is
``extern "C"``. DSL numerics wrap MLIR values; ``cutlass.Int32(result)``
wraps the call's result for the caller.

**The scratch operands are placeholders.** Nothing in Python knows
``sizeof(BlockReduce::TempStorage)`` yet. The entry point traces two
``arith.constant 0 : i32`` values, passes them as the address and size, and
records them with the enclosing kernel operation for later replacement.

Step 4: the trace-finalize hook
-------------------------------

After the kernel body and the host function have returned, CuTe calls each
registered hook with the DSL, the finished module, and the function name.
The module hash, the compile-cache lookup, and the device link all come
later, so the hook can still complete the module:

.. literalinclude:: cutlass_integration_example.py
   :language: python
   :start-after: docs: start cute-cub-finalize
   :end-before: docs: end cute-cub-finalize

The hook renders the C++ for every distinct block shape it saw, compiles it,
allocates scratch, and names the LTO-IR file on the ``gpu.module``. CuTe
merges that attribute into its ``link-libraries`` compile option and passes
the file to nvJitLink together with the kernel's own LTO-IR. The DSL
documents the attribute in terms of LLVM bitcode; its compiler accepts the
NVRTC LTO-IR container as a link input, and the link inlines the wrapper.

The C++ is two templates. The wrapper converts the 32-bit shared-window
address into a generic pointer, constructs the CUB object on it, and ends
with the barrier that makes the scratch reusable. The probe template is
explained with step 5:

.. literalinclude:: cutlass_integration_example.py
   :language: python
   :start-after: docs: start cute-cub-cpp
   :end-before: docs: end cute-cub-cpp

NVRTC compiles the translation unit as relocatable device code with
``-dlto``. The same program answers the layout query:

.. literalinclude:: cutlass_integration_example.py
   :language: python
   :start-after: docs: start cute-cub-nvrtc
   :end-before: docs: end cute-cub-nvrtc

Step 5: shared memory after the fact
------------------------------------

``nvrtcAddNameExpression`` registers the probe instantiation
``&example_layout_probe<(sizeof(T::TempStorage)), (alignof(T::TempStorage))>``
before compilation, and ``nvrtcGetLoweredName`` returns its mangled name
afterwards, for example ``_Z20example_layout_probeILy24ELy4EE``. The two
integers are the size and alignment, evaluated by the same compiler that
will lay out the struct. No device code runs and no second compilation is
needed.

With the layout known, the hook allocates the scratch at the kernel's entry
block and rewrites the traced placeholders:

.. literalinclude:: cutlass_integration_example.py
   :language: python
   :start-after: docs: start cute-cub-scratch
   :end-before: docs: end cute-cub-scratch

``SmemAllocator`` keeps no state. ``allocate`` emits a ``cute.memref.alloca``
tagged with ``smem.partition_id``; the compiler lays out all such allocations
in the kernel, including the user's own, and the host code reads the total
through ``cute.kernel_smem_size`` when it builds the launch configuration.
That is why the allocation can be inserted after tracing without touching
the launch. ``llvm.ptrtoint`` produces the 32-bit shared-window address the
wrapper expects.

The alternative, which ``cuda.coop.cutlass`` also uses for wrapper-owned
storage, is to declare ``__shared__`` storage inside the C++ wrapper and let
the device link place it. The deferred route is needed when a caller-supplied
``TempStorage`` descriptor must be shared or sliced across several calls, and
when its size must be visible to CuTe's own accounting.

The finalized module
--------------------

This excerpt is the module printed by ``CUTE_CUB_EXAMPLE_PRINT_IR=1`` on the
CUTLASS 4.9 candidate for ``sm_120a``, trimmed to the lines the integration
produced or depends on. Names and attributes vary between DSL versions.

.. code-block:: text

   gpu.module @kernels attributes {"link-libraries" = [".../92e8e7ee6bc67574.ltoir"]} {
     lir.func public @kernel_cutlass_sum_block_...(%arg0: !cute.ptr<i32, gmem, align<16>>, ...)
         attributes {cutlass_launch_facts = {exact_block_dim = array<i64: 32, 4, 1>,
                                             exact_grid_dim = array<i64: 1, 1, 1>, ...},
                     gpu.kernel, nvvm.reqntid = array<i32: 32, 4, 1>, ...} {
       // Inserted by the hook at the entry block: 24 bytes, 4-byte alignment.
       %smem = cute.memref.alloca() {smem.partition_id = 1 : i32} : !memref_smem_i8
       %iter_0 = cute.get_iter(%smem) : !memref_smem_i8
       %0 = builtin.unrealized_conversion_cast %iter_0 : !cute.ptr<i8, smem, align<4>> to !llvm.ptr<3>
       %1 = llvm.ptrtoint %0 : !llvm.ptr<3> to i32
       %c24_i32 = arith.constant 24 : i32
       ...
       %7 = cute.memref.load(%view, %coord) : (!memref_gmem_i32, !cute.coord<"?">) -> i32
       // The traced placeholders, now unused.
       %c0_i32 = arith.constant 0 : i32
       %c0_i32_8 = arith.constant 0 : i32
       // The call now carries the real address and size.
       %8 = func.call @example_block_sum_i32_32x4x1(%7, %1, %c24_i32) : (i32, i32, i32) -> i32
       ...
     }
     func.func private @example_block_sum_i32_32x4x1(i32, i32, i32) -> i32
   }
   func.func @cutlass_launch_Ptrgmem_Ptrgmem(...) attributes {llvm.emit_c_interface} {
     %0 = cute.kernel_smem_size @kernels::@kernel_cutlass_sum_block_... : i64
     ...
     %3 = cuda.launch_cfg.create<...> (blockDim = (%c32_i32, %c4_i32, %c1_i32),
                                       dynamicSmemBytes = %0, gridDim = (...), ...)
     %4 = cuda.launch_ex @kernels::@kernel_cutlass_sum_block_...<%3> (%arg0, %arg1) ...
   }

The linked cubin
----------------

Running the script with ``CUTE_DSL_KEEP=cubin`` and ``CUTE_DSL_DUMP_DIR``
set keeps the final cubin. On the same machine, ``cuobjdump --dump-sass``
shows that the device link inlined the wrapper: the listing contains no
``CALL`` and no ``example_block_sum`` symbol, one ``REDUX.SUM`` for the warp
reductions, and two ``BAR.SYNC``, one inside CUB and one from the wrapper's
trailing ``__syncthreads()``. The ELF symbol table still contains
``_Z20example_layout_probeILy24ELy4EE`` as a weak one-byte device object:
the probe instantiation survives the link. ``cuda.coop.cutlass``'s runtime
tests make the same no-call check with ``cuobjdump`` for each primitive
family.

What the package adds beyond this
---------------------------------

The script keeps one global request list and one scratch allocation per
call. ``cuda.coop.cutlass`` keeps the same five mechanisms and adds the
bookkeeping a library needs:

* **Sessions per trace.** ``BundleSession`` is keyed by the DSL's compile
  options object and the MLIR module, so nested ``@cute.jit`` calls,
  several kernels in one module, and repeated compilations stay separate.
  A snapshot restores the queued requests if a lowering fails part way.
* **Planning.** A planner selects the CUB specialization, participation
  rules, result ownership, and scratch requirements from the group, dtype,
  controls, and launch facts, and derives a deterministic symbol name from
  that plan. Equivalent requests share one wrapper.
* **Scratch policy.** ``TempStorage`` descriptors let several calls share
  one allocation or receive exclusive slices, with explicit or automatic
  reuse barriers, typed reservations, and a lifetime check across branches
  and loops. Warp primitives size one storage instance per logical warp.
* **Array results.** Load returns several values per thread through a
  ``cute.make_rmem_tensor`` buffer whose pointer is an extra argument;
  Store passes each item as a scalar argument.
* **Artifacts.** The LTO-IR is cached on disk under a key that includes the
  rendered source, header identity, compiler options, and architecture; a
  cache hit skips NVRTC and still supplies the recorded layouts. Stale
  provider paths are removed from the DSL's persistent ``LinkLibraries``
  option before the current bundle is attached.
* **Toolkit binding.** The NVRTC and nvJitLink libraries are preloaded from
  the toolkit that owns the selected headers, and the NVRTC version is
  checked before compilation.

Questions for CUTLASS reviewers
-------------------------------

1. Is ``register_trace_finalize_hook`` together with the ``gpu.module``
   ``link-libraries`` attribute the intended extension point for externally
   compiled device code? Is an NVRTC LTO-IR container, rather than LLVM
   bitcode from ``cute.ffi(source=BitCode(...))``, a supported link input
   going forward?
2. ``cute._get_launch_facts()`` is private. Is reading exact block, grid,
   and cluster shapes from the ``cutlass_launch_facts`` attribute the right
   contract, and will its schema remain versioned?
3. The hook mutates the finished module: it inserts ``SmemAllocator``
   allocations at the kernel entry block and calls
   ``replace_all_uses_with`` on traced placeholders. Is this acceptable, or
   is there a preferred way to request shared memory whose size is only
   known after an external compilation?
4. ``SmemAllocator`` is a ``dsl_user_op`` constructed inside the hook rather
   than inside the user's trace. Does that interact badly with the user's
   own allocations, partitions, or the ``smem.hint_smem_base_uniform``
   attribute?
5. The wrappers contain ``__syncthreads()`` and are inlined by the device
   link. Should the ``func.func`` prototype or call carry attributes such as
   ``convergent`` or ``noinline``, and does ``cute.ffi(inline=...)`` express
   that?
6. ``_merge_gpu_link_libraries`` appends to a compile option that persists
   across traces. The package removes its own paths before each trace.
   Would the DSL prefer to reset that option per trace itself?
7. The probe template leaves a weak one-byte ``__device__`` symbol in the
   cubin. Is there a cleaner way to obtain ``sizeof`` and ``alignof`` from
   an NVRTC compilation that CuTe will consume?
8. Are there constraints on the NVRTC version that produces the LTO-IR
   relative to the LLVM and nvJitLink bundled with the DSL? The package
   binds NVRTC to the toolkit that owns the CCCL headers and checks its
   version, but the compatibility rule itself is not documented.

Testing
-------

``tests/runtime/test_cutlass_integration_example.py`` runs the script in a
subprocess with an isolated interpreter, checks the printed verification
line, and asserts that ``cuda.coop.cutlass`` was never imported:

.. code-block:: bash

   python -m pytest -q python/cuda_coop_cutlass/tests/runtime/test_cutlass_integration_example.py
