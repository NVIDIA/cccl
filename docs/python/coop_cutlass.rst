.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-cutlass:
.. _cuda.coop.cutlass.programming_guide:
.. _cuda-coop-cutlass-cute-dsl-integration:

CUTLASS Programming Guide
=========================

Use ``cuda.coop`` inside a CuTe kernel for the cooperative primitives
documented below. The CUTLASS backend implements each supported operation
with CUB or CUDAX; see :ref:`backend coverage <coop-backends>`.

Each thread keeps its items in a ``ThreadData`` object. ``load`` fills that
object and returns ``None``; ``store`` writes its items to memory without
changing them. The examples below show the same kernel using the common API
and the CUTLASS-qualified API.

The :doc:`overview <coop>` introduces the shared concepts, installation, and
primitive families. This guide covers writing CuTe kernels; the
:doc:`CUTLASS Developer Guide <coop/cutlass_developer_guide>` explains how the
compiler integration works. For Numba kernels, see the
:doc:`Numba-CUDA-MLIR Programming Guide <coop/programming_guide>` and
:doc:`Numba-CUDA-MLIR Developer Guide <coop/developer_overview>`.

.. _coop-cutlass-api-choice:

.. _choosing-the-portable-or-qualified-api:

Choosing the common or qualified API
------------------------------------

For CUTLASS-only code, use the qualified namespace directly:

.. code-block:: python

   import cuda.coop.cutlass as coop

This import registers the CUTLASS integration. Use ordinary Load/Store and
CuTe register conversions through this one import, without a separate
``coop.register(...)`` call.

The common namespace, ``from cuda import coop``, is useful for code shared
across compilers. Examples that compare common and qualified calls use
``coop`` for the common API and ``cutlass_coop`` for the CUTLASS API. An
application can use either API on its own.

The host ``cuda.coop.register`` helper belongs to the common namespace;
qualified imports perform that registration directly.

.. list-table:: Common and CUTLASS-qualified APIs
   :header-rows: 1
   :widths: 22 38 40

   * - Feature
     - Common ``cuda.coop``
     - Qualified ``cuda.coop.cutlass``
   * - Payloads
     - Fixed per-thread ``ThreadData``; Load fills it in place.
     - Adds CuTe register-tensor and vector conversions, described in
       :ref:`coop-cutlass-register-payloads`.

.. _coop-cutlass-differences:

.. _cutlass-specific-behavior-and-current-limits:

CuTe values and supported features
----------------------------------

Use the qualified ``ThreadData`` to work with CuTe register tensors.
``ThreadData.from_register_tensor(fragment)`` copies a fragment into a
payload you can pass to ``store`` or another primitive.
``values.to_register_tensor()`` converts a payload back to a CuTe register
tensor. See :ref:`coop-cutlass-register-payloads`.

Construct payloads with ``cuda.coop.ThreadData`` or
``cuda.coop.cutlass.ThreadData`` inside a CuTe kernel. Both create CUTLASS
payloads that work with common and qualified calls. ``ThreadDataLike`` describes the shared
interface; implementing that interface in a user class does not register a
new payload representation with the compiler.


All threads in the group must call the primitive, even when ``valid_items``
selects a short tile. The sections below describe the supported groups.


.. _coop-cutlass-mixed-backends:

Mixing kernels from both compilers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CUTLASS and Numba-CUDA-MLIR kernels can run in the same process. Use the
aliases ``numba_coop`` and ``cutlass_coop`` in a module containing both:

.. code-block:: python

   import cuda.coop.numba_mlir as numba_coop
   import cuda.coop.cutlass as cutlass_coop

Call each qualified API from its own compiler's kernels. Use the
selected device's primary CUDA context before allocating memory or launching
kernels with either runtime. Numba-CUDA-MLIR requires this context; it rejects
a context created independently by another runtime.

Synchronize a kernel's work before the other runtime reads its output. Pass
data between kernels through device memory; ``ThreadData`` and CuTe register
tensors are local to the kernel that uses them.

.. _coop-cutlass-requirements:

Runtime requirements
--------------------

The base ``cuda-coop`` wheel includes this optional backend. Its initial
development target is Linux with CUDA 13. A supported public CUTLASS package
version has not yet been qualified, so ``cuda-coop`` does not provide a
CUTLASS installation extra.

The dependency-free base wheel does not install compiler prerequisites. A
CUTLASS environment also needs NumPy, ``cuda-pathfinder>=1.2.3``, and
``typing_extensions>=4.12.0`` for runtime discovery and type declarations,
alongside a compatible CuTe compiler and its dependencies.

The CuTe compiler must support linking external NVIDIA LTO-IR into a kernel,
and NVRTC must be available to compile the CUB and CUDAX functions. See the
:ref:`developer guide's compiler requirements <coop-cutlass-compiler-requirements>`
for the required CuTe integration hooks. Importing :mod:`cuda.coop` alone does
not load CUTLASS or initialize CUDA bindings.

.. _coop-cutlass-load-store:

Activation and example
----------------------

To use the common API, register CUTLASS on the host before compiling:

.. code-block:: python

   from cuda import coop

   coop.register("cutlass")

   import cutlass.cute as cute

Registration works in either import order, is safe to repeat, and remains
available when automatic registration is disabled. Importing
``cuda.coop.cutlass`` also registers the backend. For convenience, importing
``cuda.coop`` after ``cutlass`` activates it automatically.

After registration, call the primitives inside ``@cute.kernel`` or a
``@cute.jit`` function called by that kernel.

This example loads two adjacent items per thread and stores a partial tile in
the same blocked layout. ``module`` selects the common or qualified API. The
full example defines the tile dimensions and checks the output against a CPU
reference. :download:`Download the example
<../../python/cuda_coop/examples/cutlass/block_load_store.py>` to run it with a
compatible compiler:

.. literalinclude:: ../../python/cuda_coop/examples/cutlass/block_load_store.py
   :language: python
   :start-after: docs: start cutlass-block-load-store
   :end-before: docs: end cutlass-block-load-store

Block Load and Store support one-, two-, and three-dimensional blocks.
``offset`` selects the beginning of the block's tile and ``valid_items``
specifies the number of valid items in that tile. Load may fill its
out-of-bounds items with ``oob_default``. Without that default, initialize
any items that the valid prefix will not overwrite before reading them.
All threads in the block must call the primitive with uniform controls.

.. _coop-cutlass-payload-types:

Payloads, dtypes, and result ownership
--------------------------------------

``ThreadData`` holds a fixed number of values per thread. In blocked order,
thread ``t`` owns tile positions ``t * items_per_thread + i``. In striped
order, it owns positions ``t + i * group_size``. A Load/Store algorithm
determines the layout expected by that call; the payload does not carry a
layout tag. See the :doc:`Load <coop/visualizations/load>` and
:doc:`Exchange <coop/visualizations/exchange>` visualizations for the mappings.

Load infers an untyped payload's dtype from its memory operand. When building
values directly, use an explicit numeric type such as ``cutlass.Int32`` or
``numpy.int32``. Supported payload types are signed and unsigned integers of
8, 16, 32, or 64 bits and 32- or 64-bit floating point. An individual primitive
can accept a smaller set; for example, bitwise operators require integers.
Boolean, half-precision, complex, and structured payloads are unsupported.

NumPy types select a numeric representation; expressions inside the kernel
are CuTe values. Store requires the payload dtype to match the destination
element type. Cast arithmetic results explicitly when necessary, for example
with ``cutlass.Int32(value)``. Integer sums can overflow, and a parallel
floating-point sum can differ from a sequential CPU sum because the order
of additions differs.

Load initializes the destination payload in place and returns ``None``.
Store also returns ``None``. Transpose Store algorithms may rearrange the
input payload; copy values before Store if they are needed later. Other
operations document their result ownership below. Read results only at the
positions or threads where the primitive defines them.

Index payloads with compile-time integers and initialize each item before
reading it. ``ThreadData(4, dtype=cutlass.Float32, alignment=16)`` requests at
least 16-byte alignment when storage is materialized. Input and output memory
alignment is separate. The compiler decides which values remain in registers
and which spill to local memory. The :ref:`qualified conversion methods
<coop-cutlass-register-payloads>` connect payloads to CuTe register tensors
and immutable register values.

.. _coop-cutlass-helpers:

Helpers and compile-time values
-------------------------------

A ``@cute.jit`` helper called by a ``@cute.kernel`` can contain cooperative
operations. Its calls are traced in the enclosing kernel's compiler
environment and contribute to that kernel's provider bundle and scratch
requirements. Every required group member must reach a cooperative call,
including when it appears in a helper or loop.

Payload extents, group mappings, dtype selectors, algorithm names, and
``TempStorage`` constructor options must be known while tracing. Use
``cutlass.range_constexpr`` when a loop index selects payload items or
constructs different static calls. Runtime loops may repeat a fixed call
shape, and an initialized ``ThreadData`` can pass through CuTe runtime
branches and loops. Scalar controls such as ``valid_items`` and ``offset``
may be runtime values where the primitive allows them.


Block algorithms
----------------

The six block algorithms use these register layouts. For linear thread rank
``t``, item index ``i``, block size ``B``, and ``I`` items per thread, blocked
layout accesses tile index ``t * I + i``; striped layout accesses
``t + i * B``.

.. list-table:: Block Load and Store algorithms
   :header-rows: 1

   * - ``algorithm``
     - Register layout
     - Shared scratch
   * - ``direct``
     - Blocked
     - None
   * - ``striped``
     - Striped
     - None
   * - ``vectorize``
     - Blocked
     - None
   * - ``transpose``
     - Blocked
     - Required
   * - ``warp_transpose``
     - Blocked
     - Required
   * - ``warp_transpose_timesliced``
     - Blocked
     - Required

The two warp-transpose block algorithms require a block size divisible by 32.
``vectorize`` uses vector accesses when the type, item count, and address
alignment permit them, with direct accesses as a fallback.

Direct, striped, and vectorized Load/Store use no shared scratch and need no
scratch-reuse barrier, even when passed a ``TempStorage`` descriptor.
``ThreadData(alignment=...)`` requests a minimum payload alignment; it does
not change the logical item layout.

.. _coop-cutlass-storage:

Block scratch and reuse
-----------------------

Transpose algorithms allocate scratch implicitly unless passed
``temp_storage``. Construct one ``TempStorage`` inside the kernel to share
capacity across calls. An omitted size lets the compiler allocate enough
storage for all uses; an explicit byte capacity must accommodate them.
``alignment`` is a minimum: the allocation also satisfies each primitive's
alignment requirements.

Scratch belongs to one kernel execution on one block. It cannot preserve
application state between blocks or launches. CUTLASS materializes scratch
through CuTe's shared-memory allocator after tracing has collected the
requirements; CuTe owns the resulting kernel's shared-memory accounting.
See :ref:`the allocation walkthrough <coop-cutlass-scratch-allocation>` for
the compiler path. The Numba-specific ``cuda.shared.array`` coexistence rules
in the Numba guide describe that compiler's allocation model.

``sharing="shared"`` reuses one slice across call sites. With
``sharing="exclusive"``, distinct call sites receive separate slices.
Both policies default to ``auto_sync=False``. The kernel must call
``storage.sync()`` before reusing that storage, including on the next loop
iteration. Set ``auto_sync=True`` to insert trailing reuse synchronization
after each storage-using call. Without an explicit descriptor, the compiler
manages scratch and its reuse synchronization automatically.

The following example transforms eight independent tiles. It explicitly enables
automatic synchronization for a shared descriptor by default; the executable
example also supports exclusive slices and manual synchronization.
:download:`Download the storage example
<../../python/cuda_coop/examples/cutlass/block_storage.py>`:

.. literalinclude:: ../../python/cuda_coop/examples/cutlass/block_storage.py
   :language: python
   :start-after: docs: start cutlass-block-storage
   :end-before: docs: end cutlass-block-storage

Physical Warp Load and Store
----------------------------

``this_warp()`` selects the calling thread's complete 32-lane warp. The block
size must be divisible by 32, and all lanes in each participating warp must
call the primitive with uniform controls. Different warps may use different
``valid_items``, ``oob_default``, and ``offset`` values.

The four warp algorithms use the same layouts as their block counterparts:
``direct`` and ``vectorize`` use blocked layout without scratch, ``striped``
uses striped layout without scratch, and ``transpose`` uses blocked layout
with independent scratch for each warp. Transpose scratch is allocated
implicitly, with automatic warp synchronization for reuse. Explicit
``temp_storage`` is rejected for every warp algorithm.

Each warp addresses a consecutive tile within the block. For ``I`` items per
thread and linear thread rank ``t``, the compiler adds
``(t // 32) * 32 * I`` to the user-provided ``offset``. The linear rank flattens
the exact block dimensions in CUDA order: ``x + block_x * (y + block_y * z)``.
Do not add the within-block warp origin yourself. An offset for a different
block or a later loop iteration remains the caller's responsibility.

``valid_items`` counts the valid prefix of each warp's tile, from zero through
``32 * I``. As with Block Load, payload items outside that prefix are
unspecified unless ``oob_default`` is supplied, even if initialized before
Load. Store writes only the valid prefix and may rearrange its input payload
when using a transpose algorithm.

This example uses two physical warps in an ``(8, 4, 2)`` block and checks the
independent partial tiles against a CPU reference.
:download:`Download the Warp example
<../../python/cuda_coop/examples/cutlass/warp_load_store.py>`:

.. literalinclude:: ../../python/cuda_coop/examples/cutlass/warp_load_store.py
   :language: python
   :start-after: docs: start cutlass-warp-load-store
   :end-before: docs: end cutlass-warp-load-store

Logical Warp Load and Store
---------------------------

``this_warp().group_by(width)`` partitions each physical warp into consecutive
groups of 1, 2, 4, 8, 16, or 32 threads. The width and ``exhaustive`` flag must
be compile-time constants; the default exhaustive partition covers the whole
physical warp. The enclosing block must still contain complete 32-lane warps.
All four Warp Load and Store algorithms support these logical groups.

Each logical group receives its own tile and, for ``transpose``, independent
implicit scratch. With group width ``W``, linear block rank ``t``, and ``I``
items per thread, the compiler adds ``(t // W) * W * I`` to ``offset``.
Blocked layout uses tile index ``(t % W) * I + i``; striped layout uses
``(t % W) + i * W``. ``valid_items`` describes a prefix of at most ``W * I``
items. Default filling and unspecified invalid items follow the same rules
as physical Warp Load.

Every member of a participating logical group must call the primitive with
uniform controls. Complete sibling groups may take different control-flow
paths or use different offsets and valid counts. Transpose reuse
synchronization is masked to the participating logical group. Explicit
``temp_storage`` remains unsupported, and nested partitions or groups of
physical warps cannot be used for Load and Store.

This example partitions the two physical warps in an ``(8, 4, 2)`` block into
eight groups of eight threads, each loading its own partial tile.
:download:`Download the logical Warp example
<../../python/cuda_coop/examples/cutlass/logical_warp_load_store.py>`:

.. literalinclude:: ../../python/cuda_coop/examples/cutlass/logical_warp_load_store.py
   :language: python
   :start-after: docs: start cutlass-logical-warp-load-store
   :end-before: docs: end cutlass-logical-warp-load-store

.. _coop-cutlass-register-payloads:

Qualified register payloads
---------------------------

Import ``cuda.coop.cutlass`` as ``cutlass_coop`` when a kernel needs CuTe register
conversions. The qualified ``ThreadData`` provides ``from_register_tensor``
and ``to_register_tensor`` for register-memory tensors, and ``from_vector``
and ``to_tensor_ssa`` for immutable register values. These conversions use the
same fixed per-thread item count as the load/store payload.

An initialized ``ThreadData`` can cross CuTe runtime loops and branches while
retaining its fixed item count, dtype, and requested alignment. Initialize
every item in every participating thread before carrying the payload across
a runtime control-flow boundary.

Load and Store infer the memory element type from the CuTe operand. If a
producer or tensor adapter loses the intended unsigned element type, use
``cute.recast_tensor`` to restore that type before passing the tensor to
``load`` or ``store``.

.. code-block:: python

   import cuda.coop.cutlass as cutlass_coop

   # Inside a CuTe kernel, with a register-memory fragment:
   values = cutlass_coop.ThreadData.from_register_tensor(fragment)
   cutlass_coop.store(cutlass_coop.this_block(), destination, values)

.. _coop-cutlass-checking:

Checking and tuning a kernel
----------------------------

Check values and ownership against a CPU reference before timing a kernel.
Include partial tiles, nonzero offsets, multiple warp groups, and repeated
scratch reuse when those cases occur in the application. For operations with
undefined tails or nonleader results, compare only the defined outputs.

Compile before timing and synchronize the measured work. ``cute.compile``
returns a callable you can retain for repeated launches; the
:ref:`debugger walkthrough <cuda.coop.cutlass.debugger_walkthrough>` shows
compilation followed by two executions. The first compilation includes
provider generation, NVRTC, and device linking.

To inspect generated C++, set ``CUDA_COOP_SOURCE_DUMP_DIR`` before compilation.
Use the final linked cubin to assess inlining, barriers, shared memory, and
register use. A provider's source or intermediate PTX does not establish what
remains in the kernel. Use Compute Sanitizer race checking when changing
scratch reuse or synchronization.

.. _coop-cutlass-launch-facts:

Launch dimensions and resources
-------------------------------

Specify the block dimensions in the CuTe launch, including all dimensions of
a multidimensional block. Primitives specialize for those exact dimensions.
The block dimensions determine the participating threads. Group queries
and synchronization are not yet implemented by this integration. A maximum thread bound cannot substitute
for the actual participating group size. Missing required facts cause a
compilation error; see :ref:`the compiler launch contract
<coop-cutlass-exact-launch-facts>`.

More items per thread can increase register use, and additional scratch can
reduce the number of resident blocks. Check the compiled kernel's resource
usage as well as its execution time. Use inferred scratch capacity and
alignment unless the kernel needs an explicit allocation policy.
