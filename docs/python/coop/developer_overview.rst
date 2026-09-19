.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _cuda.coop.developer_overview:

``cuda.coop`` Developer Overview
================================

``cuda.coop`` makes CUB and CUDAX cooperative primitives callable inside a
Python GPU kernel. The Python compiler compiles the surrounding kernel;
``cuda.coop`` generates the C++ device functions for its primitive calls.
The two are linked together before the kernel runs.

This overview follows a call through the Numba-CUDA-MLIR implementation. It
assumes some familiarity with CUDA threads, blocks, and shared memory. The
:doc:`Programming Guide <programming_guide>` covers writing kernels and
the :doc:`overview <../coop>` covers installation and supported operations;
the focus here is how the implementation works and where to change it.
For a hands-on tour, follow the :ref:`cuda.coop.debugger_walkthrough`.

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
boundary. An application can supply device arrays instead. The primitive
sees device pointers in either case.

Each thread owns a separate ``items`` payload. With the direct algorithm,
thread 0 gets elements 0 and 1, thread 1 gets elements 2 and 3, and so on.
The Load fills that payload; the Store writes it back. This is a *blocked*
arrangement of the tile. ``ThreadData(2)`` describes two values per thread,
not two values shared by the block.

The :doc:`Load <visualizations/load>` and :doc:`Store <visualizations/store>`
visualizations show this ownership pattern and the exchanges used by other
algorithms.

All 128 threads execute both calls. There is one kernel launch. Neither
``load`` nor ``store`` launches another kernel or returns to the host.

.. _cuda.coop.calling_conventions:

Positional operands and keyword-only options
-------------------------------------------

Primitive calls take the participating group first, followed by their data
operands. These arguments are positional-only. Options such as
``algorithm``, ``valid_items``, and ``broadcast`` are keyword-only:

.. code-block:: python

   total = coop.sum(block, value)
   leader_total = coop.sum(block, value, broadcast=False)
   coop.load(block, source, items, algorithm="direct", valid_items=n)

Reduction and Scan usually need just a group and a value. Load and Store
add a source or destination. This short operand list keeps primitive
calls compact inside a kernel, while named options make choices such as
partial-tile handling and result broadcasting explicit. New optional
keyword parameters can be added without changing existing calls.

In the API reference, ``/`` marks the end of the positional-only arguments
and ``*`` introduces keyword-only parameters. For example, pass the group
and value as ``coop.sum(block, value)``, and select result broadcasting with
``broadcast=False``. With that option, only group rank zero has a defined
result; every member must still participate in the call.

``cuda.compute`` uses keyword-only parameters for all its algorithms, as
described in its :doc:`API conventions <../compute/index>`. Device-wide
algorithms can take several input and output arrays, item counts, offsets,
and a stream. Naming those arguments helps distinguish their roles and
allows callers to omit optional arguments, such as unused value buffers in
a key-only sort.

For ``cuda.coop``, the group already describes the participating threads,
and operations such as Reduction and Scan return their results directly.
The positional operands and named controls fit that smaller call shape.
When extending an API, keep the operand order consistent and use
keyword-only parameters for additional options.

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
user. That means the group shape and the kernel's other primitive calls
matter to compilation.

.. _cuda.coop.generated_shims:

Kernels and their generated C++
------------------------------

.. raw:: html

   <style>
   body { overflow-x: clip; }
   .bd-page-width { max-width: 120rem; }
   .bd-header .logo__title {
     max-width: calc(100vw - 9rem);
     overflow: hidden;
     text-overflow: ellipsis;
     white-space: nowrap;
   }
   .bd-sidebar-primary,
   .bd-sidebar-secondary { flex-basis: var(--pst-sidebar-secondary); }
   .bd-main .bd-content .bd-article-container {
     max-width: none;
     min-width: 0;
   }
   .coop-shim-pair > .sd-row {
     display: grid;
     grid-template-columns: repeat(auto-fit, minmax(min(100%, 28rem), 1fr));
   }
   .coop-shim-pair pre {
     white-space: pre-wrap;
     overflow-wrap: anywhere;
   }
   .coop-shim-pair .sd-col { width: auto; min-width: 0; }
   </style>

The following pairs use source captured while compiling real kernels. Each
kernel runs as one block of 128 threads. The copy kernels process 256
``int32`` values, with two values per thread; the Scan processes 128 values,
one per thread.

The C++ excerpts retain the emitted types, casts, and calls. Generated
identifiers have been shortened to names such as ``load_impl`` and
``load_abi``, and whitespace has been formatted to fit the page. The copy
excerpts show the no-offset Load helper and its ABI wrapper. Their full
translation units also contain Store and offset overloads. The Scan excerpt
shows the helper that accepts a scratch pointer.

The :download:`original captures <source_dumps/captures.zip>` contain all
three unmodified translation units and a manifest with their checksums and
the identifier substitutions used here. They were captured with
Numba-CUDA-MLIR 0.5.1, targeting compute capability 12.0. Generated names
and the set of emitted overloads can change with the compiler, toolkit, and
source checkout.

Capturing the source yourself
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

From the CCCL repository root, in an environment with the Numba-CUDA-MLIR
dependencies installed:

.. code-block:: bash

   export PYTHONPATH="$PWD/python/cuda_coop${PYTHONPATH:+:$PYTHONPATH}"
   export CUDA_COOP_ENABLE_CACHE=0
   export CUDA_COOP_SOURCE_DUMP_DIR="$PWD/coop-source-dumps/direct"
   python python/cuda_coop/examples/numba_mlir/source_dumps.py direct

   export CUDA_COOP_SOURCE_DUMP_DIR="$PWD/coop-source-dumps/transpose"
   python python/cuda_coop/examples/numba_mlir/source_dumps.py transpose

   export CUDA_COOP_SOURCE_DUMP_DIR="$PWD/coop-source-dumps/scan"
   python python/cuda_coop/examples/numba_mlir/source_dumps.py scan

Each command launches the selected kernel, checks its result against NumPy,
and writes ``cuda_coop_numba_mlir_<hash>.cu`` under the selected directory.
Use ``CUDA_VISIBLE_DEVICES`` as well if you need to select a particular GPU.
The :github:`example script
<python/cuda_coop/examples/numba_mlir/source_dumps.py>` includes the imports,
launches, and result checks omitted from the panels below.

Set the environment variables before starting Python. Unset or empty
``CUDA_COOP_SOURCE_DUMP_DIR`` disables dumping. The provider source is dumped
on provider-cache hits too. These
commands disable that cache and use a fresh process for each kernel so the
captures are easy to associate with their inputs. Reusing an already compiled
kernel in the same process can bypass provider generation entirely.

The dump contains the C++ input to NVRTC. Numba compiles the surrounding
kernel separately, so its indexing, launches, and planner-inserted barriers
need to be inspected in the kernel's compiler output. Bundling can also put
several providers and overloads in one source file. A definition in the dump
does not by itself establish which overload the final kernel calls.

Direct Load and Store
^^^^^^^^^^^^^^^^^^^^^

.. grid:: 1 1 2 2
   :gutter: 3
   :class-container: coop-shim-pair

   .. grid-item::

      Python kernel

      .. literalinclude:: ../../../python/cuda_coop/examples/numba_mlir/source_dumps.py
         :language: python
         :start-after: # docs: start dump-direct
         :end-before: # docs: end dump-direct

   .. grid-item::

      Generated C++: Load excerpt

      .. literalinclude:: source_dumps/direct.cpp.txt
         :language: cpp
         :start-after: // excerpt-begin

``128`` and ``2`` appear in the CUB template arguments. The ABI accepts
pointers, casts the payload pointer back to an array of two elements, and
calls ``Load``. Direct Load needs no scratch pointer. Its ``__ret`` slot is
unused because Load fills the caller's payload; the integer return value is
the ABI status. The Store wrapper uses the same pointer conversion and calls
``Store``.

Transpose with a shared scratch descriptor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. grid:: 1 1 2 2
   :gutter: 3
   :class-container: coop-shim-pair

   .. grid-item::

      Python kernel

      .. literalinclude:: ../../../python/cuda_coop/examples/numba_mlir/source_dumps.py
         :language: python
         :start-after: # docs: start dump-transpose
         :end-before: # docs: end dump-transpose

   .. grid-item::

      Generated C++: Load excerpt

      .. literalinclude:: source_dumps/transpose.cpp.txt
         :language: cpp
         :start-after: // excerpt-begin

The algorithm is now ``BLOCK_LOAD_TRANSPOSE``. CUB's ``TempStorage`` type
determines the required bytes and alignment. The ABI has an additional
pointer, ``temp_storage``, which it casts to that storage type before
calling the helper. The Python ``scratch`` descriptor causes the planner to
supply the allocation and reuse it across Load and Store.

This pointer-taking helper has no block barrier of its own. The planner
inserts the barriers in the Python kernel's lowered code. The full source
also contains ``_alloc`` variants with local ``__shared__`` storage and
``__syncthreads()``; those are separate entry points.

Scan with a Python device operator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. grid:: 1 1 2 2
   :gutter: 3
   :class-container: coop-shim-pair

   .. grid-item::

      Python kernel and operator

      .. literalinclude:: ../../../python/cuda_coop/examples/numba_mlir/source_dumps.py
         :language: python
         :start-after: # docs: start dump-scan
         :end-before: # docs: end dump-scan

   .. grid-item::

      Generated C++: Scan excerpt

      .. literalinclude:: source_dumps/scan.cpp.txt
         :language: cpp
         :start-after: // excerpt-begin

The generated source declares ``maximum_device`` and wraps its call in a
C++ lambda for CUB's ``InclusiveScan``. Numba compiles the Python
``maximum`` function into a separate LTO-IR input that supplies the declared
device symbol. Its Python body therefore has no C++ definition in this dump.

The scalar input arrives by value as ``input``. The wrapper creates
references for CUB's input and output arguments, invokes Scan, and writes
the result through ``__ret``. It still returns zero as the ABI status.
Although this Python call omits ``temp_storage``, the planner can supply
compiler-owned scratch through the same pointer-taking interface.

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
The user does not need to repeat the block size in the primitive call.
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
implementation of a primitive with a runtime group size.

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
Descriptor validation waits for default helper inlining and recursively follows
aliases and conditional definitions. A surviving unsupported helper or
descriptor escape is diagnosed with its name. Standalone callbacks cannot
contain primitives because they lack the caller's cooperative launch context.

For the MVP, ``literal_unroll`` values shaping cooperative groups, selectors,
payloads, or storage are explicitly unsupported. The planner diagnoses those
uses and suggests explicit calls with compile-time constants. Ordinary unrolling
unrelated to cooperative planning remains available. Supporting shaped unrolling
would require revisiting planner ordering; this implementation does not move
planning after SSA or unrolling.

The common API primitives in ``_core/api/`` are compiler markers with shared
signatures and validation rules. Numba's planner recognizes their identity
and binds the call arguments. Reading the Python body alone does not show
the path that runs during kernel compilation.

.. _the-portable-core-and-the-numba-backend:
.. _coop-implementation-families:

The shared core and implementation families
-------------------------------------------

The common API is exposed through ``cuda.coop`` and implemented in
``_core/api/``. The private ``_core/`` package also contains shared
implementation used by the backends. The package name describes that
implementation layer; the user-facing API is called the common API.

A :term:`family` groups related primitives and their implementation. The
Scan family, for example, has shared API declarations in
``_core/api/scan.py`` and ``scan.pyi``, semantic descriptions in
``_core/group/scan.py``, and Numba-specific entry points in
``numba_mlir/_group_scan.py`` and ``_group_scan.pyi``. Compiler analysis and
lowering have their own Scan modules. A family can span several modules
and include both common operations and qualified extensions.

The shared core describes what a primitive means and which C++ implementation
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

Payloads, layouts, and results
-----------------------------

``ThreadData`` becomes a fixed local array in Numba-CUDA-MLIR. Its extent
must be known at compile time. The compiler may keep its elements in
registers; indexing, address-taking, and register pressure determine the
final placement.

Both common and qualified ``ThreadData`` constructors accept an optional
``alignment`` keyword. It specifies a minimum power-of-two alignment in
bytes when the compiler materializes payload storage. It does not assert
alignment of the source or destination arrays passed to Load and Store.

The dtype can also come from the surrounding operation. In this kernel
fragment, the Load's source establishes the dtype:

.. code-block:: python

   items = coop.ThreadData(2)
   coop.load(block, source, items)
   coop.store(block, destination, items)

The planner propagates the dtype from ``source`` to ``items``. A later Store
or Exchange can then use ``items`` even though its constructor did not
specify a dtype. Load fills ``items`` in place and returns ``None``.

Layout describes which logical tile elements each thread owns. A striped
Load gives thread ``t`` elements ``t`` and ``t + block_size``. A blocked
Load gives it ``2 * t`` and ``2 * t + 1``. The ``transpose`` algorithm uses
striped memory transactions internally and fills the payload in blocked order;
``striped`` exposes the striped payload to the caller. The caller must
choose operations that agree on that arrangement or insert an Exchange.

The result contracts preserve the following public behavior:

* Load and Store return ``None``. Load fills the supplied output in place.
  Store preserves its input, including when its CUB implementation reorders
  data internally.
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
common payload APIs require their supported numeric dtypes.

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
inside a primitive does not generally establish that a later primitive
can immediately overwrite the same scratch.
Automatic synchronization adds a trailing barrier after each storage-consuming
call. It does not establish that arbitrary user control flow is safe: callers
must still ensure that all group members reach the primitive and barrier.

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

``sharing="exclusive"`` allocates separate slices for distinct uses. Shared
and exclusive descriptors both default to automatic synchronization; layout and
synchronization are independent. ``auto_sync=False`` leaves synchronization to
the caller. A loop that reaches the same call site again still needs safe reuse,
including with an exclusive descriptor. For a block primitive, put the required
block barrier where every thread reaches it before the next use. The planner
conservatively rejects collapsing multiple manually synchronized constructors
into one descriptor. This is a validation limit, not proof that each rejected
program races. Planner and rewrite contracts are cross-checked before emission
so parser disagreement cannot silently remove a reuse barrier.

The planner can switch its backing allocation to dynamic shared memory
when the required size exceeds the static allocation limit, subject to
the device's opt-in limit. It reports the required launch bytes through
Numba-CUDA-MLIR's compiler metadata. The user-facing call does not need a
manually maintained byte count.
The launcher treats that requirement as a minimum, not an allocation added to
user-supplied dynamic bytes. Dynamic backing accepts alignment up to 16 bytes.

Until a released compiler with the shared-memory allocation fix is qualified,
the rewrite rejects user dynamic or runtime-sized shared allocations alongside
cooperative backing, and user static shared allocations when cooperative
backing becomes dynamic. It inspects user allocations after helper inlining,
including aliases and implicit oversized cooperative scratch. Static/static
combinations remain valid, and storage-free operations introduce no conflict.
Diagnostics identify both allocations and suggest keeping them static within
the device limit, moving the user buffer to global memory, or using separate
kernels. Passing coexistence tests against a development compiler alone does
not remove the compatibility guard.

These controls are operation-specific. Warp Load/Store and Warp Scan use
compiler-owned storage and reject an explicit ``TempStorage``. Exchange
and Shuffle also manage their own scratch in the current API. CUDAX
Reduce manages scratch inside its generated C++ implementation, so an
absence of a leading scratch pointer does not mean the reduction uses no
shared memory.

Adding a Scan
-------------

With Load and Store connected, we can put a primitive between them:

.. code-block:: python

   @cuda.jit
   def scan_tile(source, destination):
       block = coop.this_block()
       items = coop.ThreadData(2, dtype=np.int32)
       coop.load(block, source, items)
       scanned = coop.exclusive_sum(block, items)
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

The :doc:`Scan visualization <visualizations/scan>` shows the ordered
prefixes and per-thread results for this operation.

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
``cuda.compute``. The operator must satisfy the primitive's mathematical
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

Importing ``cuda.coop`` after ``numba_cuda_mlir`` activates the Numba backend
hooks. An isolated common API import does not load optional compilers.
Register explicitly to make initialization independent of import order:

.. code-block:: python

   from cuda import coop

   coop.register("numba-cuda-mlir")

This host-side call imports the selected backend and activates its hooks.
It is safe to repeat. Importing ``cuda.coop.numba_mlir as numba_coop`` also
activates the hooks and exposes the backend namespace. Every install includes
the same DSL integration modules. An extra only adds dependency requirements
from ``pyproject.toml``; it does not change the wheel or register hooks in a
running process.

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

.. _cuda.coop.debugger_walkthrough:

Debugger Walkthrough
--------------------

Follow the tile copy through a Python debugger to see how the compiler
recognizes a primitive, chooses its CUB specialization, and connects the
generated device code to the kernel. These breakpoints stop in the host
Python code doing the compilation. GPU threads execute the compiled kernel;
the Python debugger cannot stop inside that device execution.

An example to debug
^^^^^^^^^^^^^^^^^^

Open ``docs/python/coop/debugger_walkthrough.py`` in your checkout, or
:download:`download the example <debugger_walkthrough.py>`. It needs no
command-line arguments:

.. literalinclude:: debugger_walkthrough.py
   :language: python
   :start-at: # Import the kernel DSL

All 128 threads copy two integers each. The second launch uses the same
argument types and block shape so you can observe compilation reuse.
Both launches check the result. Warnings about a small grid and host-array
copies are expected for this small example.

Configure VS Code
^^^^^^^^^^^^^^^^^

Use a checkout containing the Numba-CUDA-MLIR stack described above. Open
that CCCL checkout as the VS Code folder, and use **Python: Select
Interpreter** to choose an environment with the Numba-CUDA-MLIR dependencies
installed as described in the :doc:`installation instructions <../coop>`.
The Python and Python Debugger extensions must be installed in the
environment where VS Code runs the program, including the remote side
when using Remote SSH.

Add this configuration to ``.vscode/launch.json`` in the checkout. If that
file already exists, add the configuration to its ``configurations`` list:

.. code-block:: json

   {
     "version": "0.2.0",
     "configurations": [
       {
         "name": "cuda.coop: Debug active Python file",
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
           "CUDA_COOP_ENABLE_CACHE": "0",
           "CUDA_COOP_SOURCE_DUMP_DIR": "${workspaceFolder}/build/coop-debug-sources"
         }
       }
     ]
   }

``justMyCode: false`` lets you step into the compiler and library modules.
``PYTHONPATH`` and ``CUDA_COOP_CCCL_ROOT`` select this checkout's Python
sources and C++ headers. Automatic registration is enabled, and the
provider disk cache is disabled so a fresh debug session reaches NVRTC.
On a machine with several GPUs, add ``CUDA_VISIBLE_DEVICES`` to ``env``
to select the device you want to use.

With the example file active, set your **first breakpoint** on
``from cuda import coop``, using the gutter or **F9**. Select
``cuda.coop: Debug active Python file`` in **Run and Debug**, then press
**F5**. Keep the example active when starting: ``${file}`` means the
currently selected editor file.

VS Code also has a **Python Debugger: Debug Python File** editor action.
Use the named launch configuration for this walkthrough so the source
paths and library-stepping setting above apply. See the
`VS Code Python debugging documentation
<https://code.visualstudio.com/docs/python/debugging>`_ for those controls.

At a breakpoint, **F10** steps over a statement, **F11** steps into a call,
**Shift+F11** steps out, and **F5** continues to the next breakpoint.
Expressions in **Debug Console** use the selected **Call Stack** frame.
The instructions below name functions and statements so you can find the
breakpoints even as line numbers change. Put breakpoints on the executable
statement, rather than the ``def`` line or its decorator.

Initial import and registration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The first stop is just before ``cuda.coop`` imports. Numba-CUDA-MLIR has
already loaded because of the preceding import. Under
``python/cuda_coop/cuda/coop/``, open ``__init__.py`` and put a breakpoint
on ``_auto_register_known_dsls()``. Continue to it and step into the call.

In ``_core/_auto_registration.py``, follow the loop to
``candidate.activate()``. Inspect ``candidate.runtime_module`` and its
membership in ``sys.modules``: the runtime name is ``numba_cuda_mlir``.
This is why the import order matters. The automatic path recognizes a
runtime the application has already imported.

Before continuing, set a breakpoint on ``_require_runtime()`` inside
``_initialize_runtime_hooks_transaction()`` in
``numba_mlir/_compiler/_activation.py``. At this stop, the call stack
connects the root import to the qualified backend import and then to hook
registration. Step over the imports of ``_rewrite`` and ``_group_planner``;
their decorators register the compiler hooks.

For a compact confirmation, stop on ``invalid = tuple(...)`` inside
``_verify_registration_postconditions()`` in the same file. Inspect
``registration_counts``. It should contain one registration each for
``CoopGroupHierarchyPlanner``, ``CoopWholeFunctionPlanner``, and
``CoopSinglePhaseRewrite``. Registration gives Numba ways to recognize and
rewrite cooperative calls when it compiles a kernel.

Disable these import breakpoints. Set a breakpoint on the first
``copy_tile[1, 128](source, destination)`` in the example and continue.
At this point, check ``coop.__file__`` in Debug Console. It should point
inside the checkout you opened. The ``@cuda.jit`` decorator has made a
dispatcher; this first launch will trigger compilation for its arguments.

Fast-forward to the Numba hooks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Paths from here through the provider steps are relative to
``python/cuda_coop/cuda/coop/numba_mlir/``.

In ``_compiler/_group_planner.py``, find
``CoopGroupHierarchyPlanner.run()`` and set a breakpoint on
``launch_config = require_launch_config(self.state)``. Continue from the
example's launch line. You have crossed from application code into a hook
Numba calls while compiling it. Inspect the Call Stack to see that caller.

Step over the assignment, then evaluate:

.. code-block:: python

   self.state.func_ir.func_id.func_qualname
   launch_config
   self.state.func_ir.dump()

The function name is ``copy_tile``. The launch configuration contains
``block: (128, 1, 1)`` and ``grid: (1, 1, 1)``. The IR dump shows Numba's
intermediate representation of the Python function, including the
``this_block``, ``ThreadData``, ``load``, and ``store`` calls. The dump
prints to the debuggee's output; its return value is ``None``.

The block dimensions came from ``[1, 128]`` at the host launch site. They
give ``this_block()`` a concrete group shape for C++ specialization. You
are inspecting compiler values and descriptors here; ``items`` has not
become a particular GPU thread's two integers.

An earlier ``CoopSinglePhaseRewrite.match()`` can run before this stop.
It leaves group markers alone until group planning resolves them. Planner
and rewrite hooks can also run for generated helper functions. The
breakpoint above comes after the no-group-markers guard, which avoids
many uninteresting stops. This configured launch obtains its launch facts
directly; it does not require a failed first compilation to discover them.

.. _from-a-collective-call-to-a-plan:

From a primitive call to a plan
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set the next breakpoint in ``_compiler/_group_load_store.py``, inside
``_LoadStorePlanning._lower_load_store()``, on
``planned_operation = self._plan_provider_operation(plan)``. Continue.
The preceding call has produced a ``GroupLoweringPlan`` for Load.
Inspect these expressions:

.. code-block:: python

   operation
   plan.target
   dict(plan.implementation.template_arguments)
   plan.participation
   plan.temp_storage
   plan.synchronization

Expect ``operation == "load"``, target ``CUB_BLOCK``, ``T`` equal to
``int32``, ``BLOCK_DIM_X`` equal to 128, and ``ITEMS_PER_THREAD`` equal to
2. ``ALGORITHM`` selects ``::cub::BLOCK_LOAD_DIRECT``. The participation
contract requires the complete block. Direct Load needs no shared scratch,
and its storage-reuse barrier is ``NONE``.

In the same function, advance to ``statements.extend(...)`` near the end.
Compare ``factory_kwargs`` with ``runtime_args``. The block shape, dtype,
item count, and algorithm are compile-time choices in ``factory_kwargs``.
``runtime_args`` holds IR variables for ``source`` and ``items``. Those
become operands of the generated device call.

Continue to the same stops for Store. Its template selects
``BLOCK_STORE_DIRECT``, and its runtime operands are ``destination`` and
``items``. Disable the Load/Store breakpoints after inspecting both calls.
The public calls now have private provider calls carrying those choices.

Generate and compile the C++ providers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set a breakpoint on ``rewrite = CoopSinglePhaseRewrite(...)`` in
``CoopWholeFunctionPlanner.run()`` in ``_compiler/_rewrite.py``. Continue
and use ``self.state.func_ir.dump()`` again. Compare it with the earlier
dump: the group planner has introduced private factories and constants
for the resolved operations.

Before continuing, set a breakpoint in ``_types.py``, inside
``prepare_ltoir_bundle()``, on ``_, ltoir = nvrtc.compile(...)``.
At that stop, inspect ``src`` and:

.. code-block:: python

   [algo.struct_name for algo in algorithms]

This example produces both the BlockLoad and BlockStore specializations
in one C++ source unit. Find ``BLOCK_LOAD_DIRECT``, ``BLOCK_STORE_DIRECT``,
and the ``extern "C"`` ABI wrappers in ``src``. Their template arguments
should agree with the plan you just inspected. To trace an individual
provider's source generation on another run, stop in
``Algorithm._source_code()`` in the same file.

The Call Stack at the bundle stop also explains its timing:
``CoopSinglePhaseRewrite.match()`` collects function-wide storage
requirements and prepares the providers before ``apply()`` replaces the
calls. Provider compilation can supply the size and alignment facts that
storage planning needs.

Set a breakpoint in ``_compiler/_nvrtc.py``, inside ``compile_impl()``,
on ``err, prog = nvrtc.nvrtcCreateProgram(...)``. Continue and inspect
``cpp``, ``cc``, ``rdc``, ``code``, and ``compiler_options``. ``cpp`` is the
source you just saw; ``cc`` identifies your device's target architecture.
``rdc`` is true and ``code`` is ``"lto"``. The options include C++17,
the selected include directories, and ``-dlto``.

Step over ``nvrtcCompileProgram`` and the subsequent error check. NVRTC
has compiled the C++ provider code. The ``nvrtcGetLTOIR`` calls retrieve
the bytes for linking it into the Python kernel. The launch configuration
also saves the generated ``.cu`` source in ``build/coop-debug-sources``.
Open that file for a more convenient view of the complete source.

Materialize the payload and device calls
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set a breakpoint in ``CoopSinglePhaseRewrite.apply()`` in
``_compiler/_rewrite.py`` on
``self._record_invocable_specialization(invocable)``. Continue to it.
The preceding statement has materialized a callable provider for a match.
Inspect:

.. code-block:: python

   match.op_name
   match.factory_kwargs
   match.runtime_args
   invocable.files
   invocable.storage_abi
   invocable.temp_storage_bytes

Load and Store each have an ``Invocable``. Their ``files`` refer to the
shared provider LTO-IR file from the previous step. The direct algorithms
have storage ABI ``NONE`` and zero temporary-storage bytes.

After seeing both matches, disable that breakpoint and stop on
``return new_block`` at the end of ``apply()``. Evaluate
``new_block.dump()``. Find the local-array construction for ``items`` and
the calls through globals holding the ``Invocable`` objects. The local
array has extent 2 and dtype ``int32``. The group constructors have been
removed from the executable calls; dead marker references may still
appear as ``None`` pending later cleanup.

This is the point where the compiler's representation has concrete
per-thread storage and device calls in place of the public primitive
syntax. Later compilation decides whether that local array's values can
live in registers.

Hand the call and link inputs to Numba
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before continuing from ``apply()``, put a breakpoint in
``Algorithm.codegen_method()`` in ``_types.py`` on
``extern_fn = ExternFunction(...)``. At this stop, inspect ``abi_name``,
``abi_input_types``, ``arg_transforms``, and ``link_files``.

``abi_name`` matches an ABI wrapper in the generated C++. Pointer
arguments have a ``"ptr"`` transform; scalar arguments, when present,
use ``"value"``. ``link_files`` supplies the compiled provider to
Numba-CUDA-MLIR. Each specialization can expose several supported call
signatures, so this breakpoint may fire more than once per primitive.

``ExternFunction`` gives the compiler a device symbol, a signature, and
the files needed to resolve it. Its default Numba ABI includes the
status return and return-value slot shown earlier in this guide. The
Python ``Invocable.__call__`` body is a marker that rejects host calls;
kernel compilation consumes its compiler type and overload instead.

For an optional look across the dependency boundary, open the installed
``numba_cuda_mlir/mlir_lowering.py`` from the selected interpreter. Find
``lower_call_external_function()`` and stop on its call to
``self._link_external_function(fn_value)``. Inspect ``fn_value.name``,
``fn_value.sig``, ``fn_value.abi``, and ``fn_value.link``. Here the kernel
compiler receives the external function and its provider link inputs.
Step through the remainder to see it construct the MLIR call. This file
belongs to Numba-CUDA-MLIR, so use its installed source path rather than
looking for it under CCCL.

Finish the launch and observe reuse
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Disable the compiler breakpoints and set a breakpoint on the second
``copy_tile[1, 128](source, destination)`` in the example. Continue.
The first launch has completed and its assertion has passed. Inspect
``source[:8]`` and ``destination[:8]`` in the example's ``main`` frame:
``source`` starts at 0, while ``destination`` contains the ``-1`` values
written immediately before this second launch.

Re-enable the group-planner and NVRTC breakpoints, then continue. This
launch uses the existing kernel specialization, so it should finish
without those compilation stops. Both verification messages should print.

Start a new debug session to repeat the whole tour. Disabling the provider
disk cache does not disable in-process provider or kernel reuse. Editing
source while paused also does not replace the function already loaded
into that process.

A second pass: shared scratch and synchronization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Replace just the kernel in the example with this version. Before restarting,
set the two additional storage breakpoints listed below:

.. code-block:: python

   @cuda.jit
   def copy_tile(source, destination):
       block = coop.this_block()
       items = coop.ThreadData(2, dtype=np.int32)
       scratch = coop.TempStorage()
       coop.load(
           block, source, items, algorithm="transpose", temp_storage=scratch
       )
       coop.store(
           block, destination, items, algorithm="transpose", temp_storage=scratch
       )

The expected result stays the same. At the Load/Store plan breakpoint,
inspect ``plan.temp_storage`` and ``plan.synchronization`` again. The
transpose algorithms exchange data through shared memory and require a
block barrier before that scratch can be reused.

At the invocable breakpoint, the storage ABI is now ``LEADING_POINTER``, and the
temporary-storage size is nonzero. The compiled provider supplies its
size and alignment; avoid hard-coding the numbers from one toolkit.

The additional stops are in ``_compiler/_rewrite_storage.py``. Set both
before starting this pass: allocation happens before the invocable stop.

* In ``_emit_temp_storage_backing()``, stop on
  ``alloc_size = 0 if plan.uses_dynamic_smem else int(plan.total_size)``.
  Inspect ``plan.total_size``, ``plan.max_alignment``, and
  ``plan.uses_dynamic_smem``. The rewrite emits shared storage satisfying
  the provider requirements. Both uses of ``scratch`` refer to this
  planned allocation.
* In ``_emit_temp_storage_auto_sync()``, stop on ``sync_args = []``.
  Inspect ``synchronization_scope`` and ``sync_attr``. For this block
  example they select a block barrier, emitted as ``syncthreads``.

Compare ``new_block.dump()`` at the end of ``apply()`` with the direct
version. It now includes the shared allocation, scratch arguments, and
synchronization in addition to the per-thread ``items`` array.
``TempStorage`` describes shared workspace for the primitive;
``ThreadData`` holds each thread's payload. Seeing both in the rewritten
IR makes their different lifetimes and uses concrete.

If a breakpoint does not stop
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Check ``coop.__file__`` at the first launch and confirm the breakpoint
  belongs to that checkout. An older installed wheel or another worktree
  can otherwise supply the running code.
* Confirm you started the named launch configuration with
  ``justMyCode: false``. A hollow breakpoint can remain pending until its
  module loads; check it again after the import.
* If registration is skipped, restart with the imports in the example's
  order and ``CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION=0``. Read any
  automatic-registration warning for an incompatible compiler dependency.
* If NVRTC is skipped, use a fresh process with
  ``CUDA_COOP_ENABLE_CACHE=0``. Break in ``_nvrtc.compile()`` to see the
  request before the compilation cache, and check which launch you are on.

Working on a primitive
----------------------

For a new operation, start with the C++ overload and a concrete kernel
that should use it. Work out the payload arrangement, participation,
output ownership, and scratch lifetime before writing its public wrapper.
These choices determine which signatures the backend can implement.

The existing Load/Store and Scan families show the usual path:

#. Add the shared signature and type declarations in ``_core/api/`` when
   the operation belongs in the common API. Put compiler-specific
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

Result metadata must describe the returned payload independently of the
input when their shapes differ. Histogram uses ``bins_per_thread`` and a
selected counter dtype; Batched Warp Reduction returns
``ceil(batches / warp_width)`` items per thread; Discontinuity may return
one flag payload or a pair. ``GroupResultSource`` supplies dtype and extent
resolution, while the registration's ``result_resolver`` selects the
result tuple for a call. Record that information during planning so scalar
indexing and subsequent primitives can infer the result without a later
Store call supplying its type.

Keep prepared implementation state within the operation when its lifetime
does not need to cross Python calls. The bulk Run Length Decode provider
prepares a CUB run table once and uses it through an internal window loop.
Its storage contract covers the whole call. Reusing a ``TempStorage``
descriptor in a later call reuses allocation, not the prepared table.

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
     - Common signatures, descriptors, and argument rules.
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
