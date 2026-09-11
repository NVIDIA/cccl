.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _cuda.coop.developer_overview:

``cuda.coop`` Developer Overview
================================

``cuda.coop`` makes CUB and CUDAX cooperative primitives callable inside a
Python GPU kernel. The Python compiler compiles the surrounding kernel;
``cuda.coop`` generates the C++ device functions for its collective calls.
The two are linked together before the kernel runs.

This overview follows a call through the Numba-CUDA-MLIR implementation. It
assumes some familiarity with CUDA threads, blocks, and shared memory. The
:doc:`Programming Guide <programming_guide>` covers writing kernels and
the :doc:`overview <../coop>` covers installation and supported operations;
the focus here is how the implementation works and where to change it.

*Draft scope: this describes the current Numba-CUDA-MLIR 0.5.x integration,
including the Reduce and Scan work in the*
`PR stack ending at #11217 <https://github.com/NVIDIA/cccl/pull/11217>`_.
*Those changes are still under review.*

A tile copy
-----------

Start with a kernel that copies 256 integers. We launch one block of 128
threads, with two items per thread:

.. code-block:: python

   import numpy as np
   from numba_cuda_mlir import cuda

   from cuda import coop


   @cuda.jit
   def copy_tile(source, destination):
       block = coop.this_block()
       items = coop.ThreadData(2, dtype=np.int32)
       coop.load(block, source, items, algorithm="direct")
       coop.store(block, destination, items, algorithm="direct")


   source = np.arange(256, dtype=np.int32)
   destination = np.zeros_like(source)
   copy_tile[1, 128](source, destination)
   cuda.synchronize()
   np.testing.assert_array_equal(destination, source)

The example uses NumPy arrays, which Numba-CUDA-MLIR handles at the launch
boundary. An application can supply device arrays instead. The collective
sees device pointers in either case.

Each thread owns a separate ``items`` payload. With the direct algorithm,
thread 0 gets elements 0 and 1, thread 1 gets elements 2 and 3, and so on.
The Load fills that payload; the Store writes it back. This is a *blocked*
arrangement of the tile. ``ThreadData(2)`` describes two values per thread,
not two values shared by the block.

All 128 threads execute both calls. There is one kernel launch. Neither
``load`` nor ``store`` launches another kernel or returns to the host.

Calling CUB from the kernel
--------------------------

For this fixed example, the C++ work is small. The Load can be expressed as:

.. code-block:: c++

   #include <cub/block/block_load.cuh>

   __device__ void load_tile(const int* source, int (&items)[2])
   {
     using block_load = cub::BlockLoad<int, 128, 2, cub::BLOCK_LOAD_DIRECT>;
     block_load{}.Load(source, items);
   }

This function is called by every thread in the block. CUB uses the thread
index to decide which elements that thread should load. The block size,
item count, dtype, and algorithm are C++ template arguments.

The Python compiler needs a callable device symbol with a known calling
convention. It cannot call a C++ template directly, and its array value is
not a C++ array reference. We therefore generate a wrapper that exposes the
specialization through the compiler's device calling convention (ABI).
For this Load, a simplified wrapper looks like:

.. code-block:: c++

   extern "C" __device__ int load_tile_abi(
     void* result, void* source, void* items)
   {
     load_tile(
       static_cast<const int*>(source),
       *reinterpret_cast<int (*)[2]>(items));
     return 0;
   }

Here the integer return is the ABI status. The leading pointer is the
return-value slot and is unused for this void operation. The remaining
pointers address the input and this thread's output payload. For an
operation returning a scalar, the wrapper writes that scalar through the
return-value pointer. Scalar inputs can be passed by value; arrays and
scratch storage use pointers.

These snippets explain the wrapper rather than reproduce its generated
spelling. The real wrapper names encode the specialization and compilation
identity, and the generated source includes the required support types.
``Algorithm._source_code()`` and ``Algorithm._emit_abi_wrapper()`` in
``numba_mlir/_types.py`` generate this code.

The backend calls this implementation of the primitive a *provider*.
NVRTC compiles its generated C++ as relocatable device code using C++17 and
``-dlto``. The result is LTO-IR, the intermediate representation used for
link-time optimization. An ``Invocable`` records the callable signature,
link inputs, and any scratch requirements. Its compiler type uses
Numba-CUDA-MLIR's ``ExternFunction`` to represent the external call.

Numba-CUDA-MLIR compiles the Python kernel and supplies the provider's
LTO-IR to the device link. Link-time optimization can inline the CUB wrapper
into the kernel and optimize across that call. Inlining and register use
still depend on the resulting code; LTO is not a guarantee about either.
Numba-CUDA-MLIR owns the final kernel compilation, loading, and launch.

Compared with the :doc:`cuda.compute overview <../compute/developer_overview>`,
the same runtime compilation tools appear at a different boundary. Here
the generated C++ implements a device call within a kernel supplied by the
user. That means the group shape and the kernel's other collective calls
matter to compilation.

Recovering the specialization
----------------------------

The fixed C++ example supplied all its template arguments by hand. In the
Python kernel, some of that information is in the call, some comes from
type inference, and some comes from the configured launch:

.. list-table::
   :header-rows: 1
   :widths: 27 35 38

   * - Information
     - Source in the example
     - Use
   * - Group
     - ``this_block()`` and ``[1, 128]``
     - Resolve a block with dimensions ``(128, 1, 1)``.
   * - Payload dtype
     - ``ThreadData(..., dtype=np.int32)``
     - Select the C++ element type and check the source and destination types.
   * - Items per thread
     - ``ThreadData(2, ...)``
     - Instantiate a fixed array extent of two.
   * - Algorithm
     - ``algorithm="direct"``
     - Select the CUB algorithm and its storage requirements.
   * - Input and output addresses
     - Kernel arguments
     - Pass device pointers when the kernel executes.
   * - Target architecture
     - Compiler target
     - Compile compatible device code for the surrounding kernel.

``this_block()`` is a compile-time group descriptor. The planner resolves
it against the launch and removes the descriptor from the runtime code.
The user does not need to repeat the block size in the collective call.
Launching the same kernel with a different block shape can require a
different specialization.

The full shape matters. A block of ``(32, 4, 1)`` and a block of
``(128, 1, 1)`` both contain 128 threads, but their coordinate expressions
differ. Group indexing uses the x-major linear rank:

.. code-block:: text

   threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)

Groups can also describe partitions. ``this_warp().group_by(8)`` describes
four consecutive groups of eight threads within a physical warp. The
partition width is a compile-time value. Each group's rank, memory tile,
scratch instance, and synchronization mask must agree on that partition.
For Warp Load/Store, the enclosing block must contain complete physical
warps, and every member of a participating logical group must reach the
call.

Group methods such as ``rank()`` and ``count()`` produce integer values
that the kernel can use. The group descriptor itself remains compile-time
information. Adding a descriptor or query for a scope does not supply an
implementation of a collective with a runtime group size.

There is also a distinction between a static group size and a runtime
quantity measured within that group. A tail Load may use:

.. code-block:: python

   # Inside a kernel, after clamping remaining to this block's tile size:
   coop.load(block, source, items, valid_items=remaining, oob_default=0)

The CUB specialization still describes the full tile. ``remaining`` says
how many elements in that tile are valid on this invocation. It can be a
runtime value, provided it is uniform across the group and lies between
zero and the tile capacity.

The core represents optional arguments with ``ArgumentBinding``:

* ``OMITTED`` selects the overload without that argument.
* ``STATIC`` carries a compile-time value into specialization and code
  generation.
* ``RUNTIME`` keeps an operand in the device call.

Those cases affect the ABI and cache identity. Treating a runtime value as
a constant would compile the wrong program; treating an omitted argument
as zero could select a different CUB overload. The planner rejects invalid
static controls before compilation. For bounded runtime controls such as
``valid_items``, the generated code checks the range before narrowing to
CUB's integer parameter. A failed check traps on the device. Callers must
also provide enough memory for the selected tile and offset.

From Python syntax to an external call
-------------------------------------

The integration uses two whole-function planners and a before-inference
rewrite to inspect and modify the compiler's intermediate representation
(IR). Their implementation names appear below because they are useful
places to start reading the code.

#. ``CoopGroupHierarchyPlanner`` finds group constructors, group methods,
   and group-first primitive calls. It requests exact launch metadata,
   resolves descriptors and payload information, and asks the primitive
   family's core planner for a lowering plan.
#. The family's Numba implementation turns that plan into private provider
   calls. At this point, the operation and its C++ specialization are known,
   but the kernel still needs concrete payloads, storage, and callables.
#. ``CoopSinglePhaseRewrite`` materializes those calls. It replaces
   ``ThreadData`` with fixed local arrays, builds invocables, supplies
   scratch pointers where needed, and emits result handling and reuse
   barriers. It removes the compile-time constructors from the runtime IR.
#. ``CoopWholeFunctionPlanner`` gives the rewrite a whole-function retry
   after launch information and inlined device helpers are available.
   Compilation then continues with ordinary typed device calls.

These are responsibilities rather than a claim that every kernel takes
four passes in that exact order. Numba-CUDA-MLIR can retry compilation when
a planner requests launch facts. The initial rewrite leaves unresolved
group calls in place until the group planner can handle them.

Inlining is relevant to helper functions. A helper that receives a group
and calls ``coop.load`` can be planned after it is inlined into its kernel
caller. The descriptor then has the caller's launch context. A group
descriptor escaping into an arbitrary runtime object or a non-inlined
device call is not supported by this mechanism.

The portable functions in ``_core/api/`` are compiler markers with shared
signatures and validation rules. Numba's planner recognizes their identity
and binds the call arguments. Reading the Python body alone does not show
the path that runs during kernel compilation.

The portable core and the Numba backend
--------------------------------------

The core describes what a collective means and which C++ implementation
can perform it. It does not import Numba or invoke a compiler. The Numba
backend reads compiler IR and types, then converts the core's plan into
code that Numba-CUDA-MLIR can compile.

For example, the Load/Store family normalizes the group, algorithm, dtype,
item count, and optional arguments into a ``GroupPrimitiveCall``.
``plan_group_primitive()`` resolves that call to a ``GroupLoweringPlan``.
The plan records more than the selected CUB class:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Contract
     - What the backend needs to know
   * - Topology and participation
     - Which threads form a group, how many group instances exist in a
       block, and which participants must reach the call.
   * - Result
     - Whether the result aliases an input, needs new storage, or is a
       scalar; which threads have a defined result.
   * - Temporary storage
     - Who owns scratch, how many instances are needed, and which layout
       requirements must be obtained from the compiled provider.
   * - Synchronization
     - Which barrier is required before another call can reuse scratch.
   * - Implementation
     - The CUB specialization or CUDAX call, along with its source library
       and header.

A supported plan must contain the required contracts. An unsupported
combination carries a reason that the backend reports before provider
compilation. For instance, having a ``ThreadGroup`` descriptor for a scope
does not imply that every primitive supports that scope.

``AlgorithmSpec`` describes a CUB template specialization and its method
parameters without compiler types. ``NumbaMlirCoreAdapter`` maps those
types and parameters to the Numba backend's representation. CUDAX group
calls use a separate call description and generated wrapper.

Reduce illustrates why implementation selection belongs in the family
planner. Full reductions with supported built-in operators use CUDAX's
hierarchy-aware implementation. Prefix reductions, explicit CUB algorithm
selection, and qualified custom operators take supported CUB paths. The
same public operation can therefore have different implementation and
storage contracts depending on its arguments.

*The current stack still marks mapped warps-within-block scalar Reduce as
an expected failure pending the separate*
`CUDAX scratch-reuse fix <https://github.com/NVIDIA/cccl/pull/10985>`_.

Payloads, layouts, and results
-----------------------------

``ThreadData`` becomes a fixed local array in Numba-CUDA-MLIR. Its extent
must be known at compile time. The compiler may keep its elements in
registers; indexing, address-taking, and register pressure determine the
final placement.

Both portable and qualified ``ThreadData`` constructors accept an optional
``alignment`` keyword. It specifies a minimum power-of-two alignment in
bytes when the compiler materializes payload storage. It does not assert
alignment of the source or destination arrays passed to Load and Store.

The dtype can also come from the surrounding operation. In this kernel
fragment, the Load's source establishes the dtype:

.. code-block:: python

   items = coop.ThreadData(2)
   loaded = coop.load(block, source, items)
   coop.store(block, destination, loaded)

The planner must track both the result's type source and its alias. Here
``loaded`` refers to ``items``, while its dtype comes from ``source``. That
distinction lets a later Store or Exchange use the result of a Load whose
payload did not have an explicit dtype.

Layout describes which logical tile elements each thread owns. A striped
Load gives thread ``t`` elements ``t`` and ``t + block_size``. A blocked
Load gives it ``2 * t`` and ``2 * t + 1``. The ``transpose`` algorithm uses
striped memory transactions internally and returns a blocked payload;
``striped`` exposes the striped payload to the caller. The caller must
choose operations that agree on that arrangement or insert an Exchange.

The result contracts preserve the following public behavior:

* Load fills the supplied output and returns that same payload. Store
  returns ``None`` and preserves its input, including when its CUB
  implementation reorders data internally.
* Exchange and array Scan return a fresh payload. Their inputs remain
  available to subsequent kernel code.
* Reduce returns a scalar. The default ``broadcast=True`` makes the result
  available throughout the group. With ``broadcast=False``, only group
  rank zero has a defined result, although every required thread must
  still participate.

Output ownership is part of lowering. A CUB method that overwrites an
array does not, by itself, implement a Python operation that promises to
preserve that array. The backend may need a copy or a separate result
payload around the provider call.

The qualified namespace accepts additional compiler-specific values, such
as local-array payloads where supported. Type support is still checked by
each primitive. An ABI helper for aggregate values does not imply that
public Load, Reduce, or Scan accepts arbitrary structures. The current
portable payload APIs require their supported numeric dtypes.

Shared memory and reuse
-----------------------

The direct tile copy needs no CUB scratch. Direct, striped, and vectorize
Load/Store providers have ``StorageABI.NONE`` and do not add a scratch
pointer or a reuse barrier.

Changing the Load to ``algorithm="transpose"`` introduces shared-memory
communication. CUB defines a ``TempStorage`` type whose size and alignment
depend on the specialization. ``cuda.coop`` obtains both from compiled
code: generated globals contain ``sizeof`` and ``alignof`` values, and a
metadata link to PTX makes those constants available to the host planner.
The provider's LTO-IR is retained for the final kernel link. The compiler
does not run a GPU kernel to discover the scratch layout.

Storage-bearing CUB providers use ``StorageABI.LEADING_POINTER``. The
rewrite supplies a pointer to the appropriate slice of the kernel's
shared-memory allocation. This is the first *provider operand*; the
external ABI's return-value slot is separate.

The planner considers uses across the function so compatible sequential
calls can reuse storage. It also keeps independent group instances
separate. A transpose in each physical warp needs a distinct scratch
slice for each warp; logical warps need slices and synchronization masks
for their smaller groups. Equal byte counts do not make storage from
different participation domains interchangeable.

After a storage-bearing block call, the rewrite normally emits a block
reuse barrier. For supported physical and logical Warp calls, it emits
``syncwarp`` with the participating group's mask. CUB's synchronization
inside a collective does not generally establish that a later collective
can immediately overwrite the same scratch.

A block operation can expose that reuse choice through ``TempStorage``:

.. code-block:: python

   # Inside a kernel; source and destination each contain one full tile.
   scratch = coop.TempStorage()
   items = coop.ThreadData(2, dtype=np.int32)
   coop.load(block, source, items,
             algorithm="transpose", temp_storage=scratch)
   coop.store(block, destination, items,
              algorithm="transpose", temp_storage=scratch)

The default shared descriptor permits reuse and automatic barriers. Its
size and alignment can be inferred from its uses. An explicit capacity
must satisfy the compiled provider's requirements.

Only ``size_in_bytes`` may be positional in ``TempStorage``; the other
options are keyword-only. An explicit ``alignment`` requests a minimum,
which the planner can strengthen to meet the requirements of its uses.

``sharing="exclusive"`` allocates separate slices for distinct uses and
disables automatic reuse synchronization. ``auto_sync=False`` on shared
storage leaves synchronization to the caller. A loop that reaches the
same call site again still needs safe reuse, including with an exclusive
descriptor. For a block collective, put the required block barrier where
every thread reaches it before the next use.

The planner can switch its backing allocation to dynamic shared memory
when the required size exceeds the static allocation limit, subject to
the device's opt-in limit. It reports the required launch bytes through
Numba-CUDA-MLIR's compiler metadata. The user-facing call does not need a
manually maintained byte count.

These controls are operation-specific. Warp Load/Store and Warp Scan use
compiler-owned storage and reject an explicit ``TempStorage``. Exchange
and Shuffle also manage their own scratch in the current API. CUDAX
Reduce manages scratch inside its generated C++ implementation, so an
absence of a leading scratch pointer does not mean the reduction uses no
shared memory.

Adding a Scan
-------------

With Load and Store connected, we can put a collective between them:

.. code-block:: python

   @cuda.jit
   def scan_tile(source, destination):
       block = coop.this_block()
       items = coop.ThreadData(2, dtype=np.int32)
       loaded = coop.load(block, source, items)
       scanned = coop.exclusive_sum(block, loaded)
       coop.store(block, destination, scanned)


   source = np.arange(256, dtype=np.int32)
   destination = np.zeros_like(source)
   scan_tile[1, 128](source, destination)
   cuda.synchronize()

   expected = np.zeros_like(source)
   expected[1:] = np.cumsum(source[:-1], dtype=np.int32)
   np.testing.assert_array_equal(destination, expected)

The blocked arrangement defines the scan order across the tile. The
result is a new two-item payload for each thread. Block Scan uses CUB
temporary storage even though this example's Load and Store do not.

This computes one block's prefix sum. Processing multiple blocks requires
the caller to assign separate tiles and, for a device-wide scan, arrange
the carry between them. Merely increasing the grid size does not do that.

Python operators and prefix state
---------------------------------

A built-in sum can use a C++ operator directly. A Python-defined operator
adds another compilation input. This example uses the qualified namespace,
which supports stateless Python Scan operators:

.. code-block:: python

   import cuda.coop.numba_mlir as numba_coop


   @cuda.jit(device=True)
   def maximum(lhs, rhs):
       return lhs if lhs > rhs else rhs


   @cuda.jit
   def prefix_maximum(source, destination):
       block = numba_coop.this_block()
       value = source[cuda.threadIdx.x]
       result = numba_coop.inclusive_scan(block, value, scan_op=maximum)
       destination[cuda.threadIdx.x] = result


   source = np.array([3, 1, 4, 2] * 32, dtype=np.int32)
   destination = np.zeros_like(source)
   prefix_maximum[1, 128](source, destination)
   cuda.synchronize()
   np.testing.assert_array_equal(destination, np.maximum.accumulate(source))

Once the operand dtype is known, the backend compiles ``maximum`` with
``cuda.compile(..., output="ltoir", cc=...)`` for the same target as the
provider. The generated C++ declares its device symbol and creates a C++
callable that invokes it. CUB receives that callable as its scan operator.
The operator's LTO-IR joins the provider's link inputs. No Python callback
runs while the GPU executes the scan.

This uses the same general technique as Python operators in
``cuda.compute``. The operator must satisfy the collective's mathematical
contract, including associativity, and the supported input and output
dtype contract. Successful compilation cannot establish associativity.

A Block Scan *prefix callback* is a different operand from its binary scan
operator. It receives the block aggregate and returns a prefix to apply
to that block's scan. The qualified API names it ``prefix_op``. For a
running prefix, ``StatefulFunction`` describes a callback whose first
argument is a pointer to explicit per-thread state; the state payload is
passed as the third positional argument to the Scan call.

The descriptor identifies the callable and state dtype. The state itself
remains runtime data, so repeated calls can update it without changing the
compiled operator. Its lifetime is separate from CUB scratch reuse. The
current implementation requires numeric, one-item state and exact dtype
matching with the descriptor. The state dtype can differ from the scan
dtype.

CUB invokes the prefix callback in the first warp of the block. Only
lane zero's returned prefix is used, and only thread zero's state is
authoritative after the call. Callers initialize each participating state
cell equally. This is local state for successive tiles handled by one
block; it does not provide communication between blocks. Prefix callbacks
currently cannot be combined with ``initial_value`` or
``aggregate_output`` and are not supported for Warp Scan.

Activation and compilation reuse
--------------------------------

The import order in the first example is intentional. Importing
``cuda.coop`` after ``numba_cuda_mlir`` activates the Numba backend hooks.
An isolated portable import does not load optional compilers. If the
portable module was imported first, an explicit qualified import activates
the hooks:

.. code-block:: python

   import cuda.coop.numba_mlir as numba_coop

Use an alias: a bare dotted import would bind ``cuda`` to the top-level
package and could replace the local name used for Numba's ``cuda.jit``.

``_compiler/_activation.py`` registers the planners and rewrite.
``_compiler/_numba_mlir_compat.py`` isolates access to Numba-CUDA-MLIR's
private overload, IR, datamodel, and registry APIs. It supports one runtime
series without adapting between versions. The package currently bounds
that dependency to ``>=0.5.0,<0.6``. A new compiler series needs its
integration checked before that bound changes.

The rewrite can collect compatible CUB specializations and compile them
in a single NVRTC source bundle. This reduces repeated header parsing and
compilation. Each call still has its own signature, result handling, and
storage contract.

There are caches at several stages. Compiled Python operators are reused
within the process. Provider compilation can use a persistent cache when
``CUDA_COOP_ENABLE_CACHE`` is enabled before backend import. Numba-CUDA-MLIR
separately owns the compiled kernel's reuse and lifetime. The provider cache
uses ``XDG_CACHE_HOME/cccl`` on POSIX, falling back to ``~/.cache/cccl``;
on Windows it uses ``LOCALAPPDATA\cccl``, falling back to
``~\AppData\Local\cccl``. Unset, empty, or relative base directories use
the fallback. These settings are read at backend cache import.

The provider cache uses ``$XDG_CACHE_HOME/cccl`` on POSIX systems, falling
back to ``~/.cache/cccl``. On Windows it uses ``%LOCALAPPDATA%\cccl``, with
``~\AppData\Local\cccl`` as the fallback. Cache configuration is read when
the backend cache module is imported.

A provider cache key must identify the code being compiled: the operation,
dtype, shape, static arguments, wrapper ABI, target architecture, compiler
options, and selected headers and compiler libraries. Runtime array
addresses and runtime tail counts are operands, not specialization values.
The header and toolkit identity matters because the same CUB template
arguments can produce different code or scratch layouts with different
headers.

The ``cuda-coop`` wheel bundles matching CCCL headers. A source checkout
uses its checkout's headers, and ``CUDA_COOP_CCCL_ROOT`` can select an
explicit root. NVRTC, its builtins, nvJitLink, and CUDA headers must also
resolve coherently. ``_headers/`` and ``_compiler/_nvrtc.py`` handle that
selection. An invocable retains its temporary LTO-IR files for linking;
those files are compilation inputs, not loaded kernel handles.

Working on a primitive
----------------------

For a new operation, start with the C++ overload and a concrete kernel
that should use it. Work out the payload arrangement, participation,
output ownership, and scratch lifetime before writing its public wrapper.
These choices determine which signatures the backend can implement.

The existing Load/Store and Scan families show the usual path:

#. Add the shared signature and type declarations in ``_core/api/`` when
   the operation belongs in the portable API. Put compiler-specific
   extensions in the qualified namespace.
#. Describe the operation and C++ overload in the core family. Its group
   planner selects a supported implementation and returns complete result,
   topology, storage, and synchronization contracts.
#. Add the Numba family binding under ``numba_mlir/_compiler/`` and its
   provider lowering under ``numba_mlir/_lowering/``.
   ``register_group_primitive()`` connects group-call planning;
   ``register_rewrite_operation()`` describes the provider rewrite's
   arguments and family-specific analysis.
#. Check that the existing materialization and storage code can consume
   that description. Change the shared rewrite only when the new operation
   needs a compiler behavior that the existing contracts cannot express.

Tests live under ``python/cuda_coop/tests/``. Core contract tests check
normalization and plan selection without a compiler. Backend unit tests
check argument binding, inference, rewrites, and diagnostics. Compile
tests use real NVRTC and nvJitLink with devices hidden; their fixtures
provide an explicit target. Runtime tests check the resulting kernels.

Use tests that exercise the part you changed. A result-ownership change
needs an input-preservation check. A storage change needs repeated calls
and multiple independent groups. A callable ABI change needs a real link
and a runtime result. A mocked compiler test cannot establish that the
generated wrapper and operator agree on their ABI.

For example, from the CCCL root in an environment with the Numba backend
and test dependencies installed:

.. code-block:: console

   CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD/python/cuda_coop" python -m pytest \
     python/cuda_coop/tests/contracts/core/test_core_group_load_store.py \
     python/cuda_coop/tests/backends/numba_mlir/unit/test_group_lowering_plan.py

   CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD/python/cuda_coop" python -m pytest \
     python/cuda_coop/tests/backends/numba_mlir/compile/test_block_load_store_compile.py

The source path prevents an older installed ``cuda-coop`` wheel from
silently supplying the module under test. The compile tests' fixed target
is a test fixture; it does not imply that every public kernel compilation
path is available without a GPU or configured launch.

To inspect generated C++, set ``CUDA_COOP_SOURCE_DUMP_DIR`` to a
directory before compiling the kernel. Files are named
``cuda_coop_<backend>_<hash>.cu``, allowing backends to share the directory.
The dump is useful for checking
template arguments, wrapper signatures, and scratch metadata. Inspect
the final kernel's PTX or SASS separately for inlining, barriers, and
register behavior.

Source map
----------

Paths below are relative to ``python/cuda_coop/cuda/coop/``:

.. list-table::
   :header-rows: 1
   :widths: 47 53

   * - Path
     - Responsibility
   * - ``_core/api/`` and the adjacent ``.pyi`` files
     - Portable signatures, descriptors, and argument rules.
   * - ``_core/group/``
     - Group resolution, primitive semantics, and lowering contracts.
   * - ``_core/block/`` and ``_core/warp/``
     - CUB algorithm specifications and generated support code.
   * - ``numba_mlir/_compiler/_group_planner.py`` and ``_group_*.py``
     - Recover group calls from compiler IR and invoke family planners.
   * - ``numba_mlir/_compiler/_rewrite.py`` and ``_rewrite_*.py``
     - Materialize payloads, invocables, result handling, and storage.
   * - ``numba_mlir/_lowering/``
     - Translate core specifications and generate CUB or CUDAX providers.
   * - ``numba_mlir/_types.py``
     - Device ABIs, Python operator compilation, source generation, and
       invocable link inputs.
   * - ``_headers/`` and ``numba_mlir/_compiler/_nvrtc.py``
     - Header and toolkit selection, compile identity, and NVRTC invocation.

Runnable kernels are in ``python/cuda_coop/examples/numba_mlir/``. The
installed ``.pyi`` files and :doc:`API reference <../coop_api>` describe
the public signatures; private provider factories are implementation
details.
