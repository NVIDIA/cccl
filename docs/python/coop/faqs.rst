.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-faqs:

FAQs
====

.. _coop-faq-namespaces:
.. _why-are-there-both-cuda-coop-and-cuda-coop-numba-mlir:

.. _why-are-there-portable-and-backend-qualified-namespaces:

Why are there common and backend-qualified namespaces?
------------------------------------------------------

``cuda.coop`` provides the contract shared by Numba-CUDA-MLIR and CUTLASS.
Both backends implement its kernel operations. Use this namespace for code
that shares group, ``ThreadData``, and built-in operator contracts across
compilers. Import your kernel compiler before ``cuda.coop`` so its backend
is registered automatically:

.. code-block:: python

   from numba_cuda_mlir import cuda

   from cuda import coop

For CuTe kernels:

.. code-block:: python

   from cutlass import cute

   from cuda import coop

These examples need no explicit ``coop.register`` call. If you cannot
ensure import order, see :ref:`when to register explicitly
<coop-faq-register>`.

The qualified namespaces, ``cuda.coop.numba_mlir`` and
``cuda.coop.cutlass``, each include all common kernel operations and add
features specific to their compiler. Numba's extensions include local-array
payloads and device callbacks. CUTLASS adds CuTe register-tensor conversions.
Both add operation-specific controls;
see the :ref:`Numba comparison <coop-programming-api-choice>` and
:ref:`CUTLASS comparison <coop-cutlass-api-choice>`.

Importing a qualified namespace also registers its backend. Common and
qualified calls for the same compiler can appear in one kernel and follow
the shared contracts. Kernel launch syntax and other DSL code still need
adaptation when moving between compilers; compiler-owned payloads cannot
cross that boundary. The :doc:`API reference <../coop_api>` lists the
common operations and each compiler's extensions.

.. _i-only-use-numba-cuda-mlir-can-i-import-its-namespace-as-coop:
.. _coop-faq-numba-only:
.. _coop-faq-qualified-only:

Can I use only a qualified namespace?
-------------------------------------

Yes. Import the qualified namespace for your kernel compiler:

.. code-block:: python

   from numba_cuda_mlir import cuda

   import cuda.coop.numba_mlir as coop

For CuTe kernels:

.. code-block:: python

   from cutlass import cute

   import cuda.coop.cutlass as coop

Each import registers its backend, so these examples need no separate
``register`` call. The host ``cuda.coop.register`` helper belongs to the
common namespace; qualified imports perform that registration directly.

Use ``numba_coop`` and ``cutlass_coop`` when a module contains both DSLs,
and call each API from its own compiler's kernels. Examples and shared
helpers may also use the common ``coop`` API alongside a qualified import.
The documentation uses the longer aliases to make those comparisons clear;
a single-backend application can use ``coop`` throughout.

Keep the alias on a dotted import. Bare ``import cuda.coop.numba_mlir``
assigns the top-level package to ``cuda`` in that scope, replacing the name
previously imported from ``numba_cuda_mlir``.

.. _coop-faq-register:

When do I need to use ``coop.register("<dsl>")``?
------------------------------------------------------------

With automatic registration enabled, you only need an explicit
``coop.register`` call when using the common API if you cannot ensure that
the compiler is imported before the first ``cuda.coop`` import. The
compiler-first examples above register the backend automatically.

Notebooks are a common example: cells can be run out of order or rerun,
and imports from earlier cells remain cached. If an earlier cell or an
imported library loaded ``cuda.coop`` before the compiler, rerunning the
imports in the right order does not rerun ``cuda.coop``'s import-time
registration. Put an explicit call in your setup cell before compiling
kernels:

.. code-block:: python

   from cuda import coop
   from numba_cuda_mlir import cuda

   coop.register("numba-cuda-mlir")

Use ``coop.register("cutlass")`` for CuTe kernels. There is no harm in
calling ``register`` when the backend is already registered, or in rerunning
the setup cell. Repeated calls are safe; you do not need to check the import
order or whether registration has already happened before making the call.

Importing a qualified namespace such as ``cuda.coop.numba_mlir`` also
registers its backend, so it needs no separate call. If you deliberately
disable automatic registration with
``CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION=1``, use explicit registration or a
qualified import regardless of import order. See :ref:`backend registration
<coop-backend-registration>` for supported names and setup requirements.

.. _coop-faq-thread-data-local-array:

Why do I need to use ``coop.ThreadData``? Why can't I use ``cuda.local.array``?
--------------------------------------------------------------------------------

`cuda.local.array
<https://nvidia.github.io/numba-cuda-mlir/latest/user/memory.html#local-memory>`_,
a Numba CUDA function, allocates an array private to each kernel thread.

Its size and any explicit alignment must be compile-time constants, and its
dtype must be known when the kernel is specialized. An array size computed
from values read during kernel execution is unsupported. These two kernels
show the difference:

.. code-block:: python

   import numpy as np
   from numba_cuda_mlir import cuda

   @cuda.jit
   def kernel1(src, dst):
       # Valid: the size is constant and the dtype is explicit.
       items = cuda.local.array(2, dtype=np.int32)
       offset = cuda.threadIdx.x * 2
       for i in range(2):
           items[i] = src[offset + i]
       for i in range(2):
           dst[offset + i] = items[i]

   @cuda.jit
   def kernel2(src, dst):
       # Invalid: this size depends on values read during execution.
       size = src.size // cuda.blockDim.x
       # src.dtype is supported; the runtime size is the problem here.
       items = cuda.local.array(size, dtype=src.dtype)
       offset = cuda.threadIdx.x * size
       for i in range(size):
           items[i] = src[offset + i]
       for i in range(size):
           dst[offset + i] = items[i]

   # These examples copy one full block's worth of data, two items per thread.
   threads_per_block = 32
   dtype = np.int32
   h_src = np.arange(threads_per_block * 2, dtype=dtype)
   h_dst = np.empty_like(h_src)
   d_src = cuda.to_device(h_src)
   d_dst = cuda.device_array_like(d_src)

   kernel1[1, threads_per_block](d_src, d_dst)
   d_dst.copy_to_host(h_dst)
   np.testing.assert_array_equal(h_dst, h_src)

The ``cuda.local.array`` in ``kernel1`` is valid because the size and dtype
are known at compile time. The loops copy the values directly so this is a
working local-array example; the common ``coop.load`` and ``coop.store``
calls below use ``coop.ThreadData`` for their array payloads.

Attempting to launch ``kernel2`` raises a typing error during compilation,
before the kernel runs:

.. code-block:: python

   kernel2[1, threads_per_block](d_src, d_dst)  # Raises a typing error.

The restriction is on runtime values, not just on how an expression is
written. Constant expressions and captured host values can also supply
the size. Likewise, ``dtype=src.dtype`` works in Numba-CUDA-MLIR because
the source array's element type is known when the kernel is specialized.
``cuda.shared.array``, the shared-memory counterpart, also supports dynamic
shared memory; it does not have exactly the same size restrictions.

A kernel factory can compute the size on the host and capture it with the
dtype. Numba-CUDA-MLIR then sees those captured values as compile-time
constants:

.. code-block:: python

   def make_kernel1(num_elems, threads_per_block, dtype):
       # This example assumes one full block and an equal share per thread.
       if num_elems <= 0 or threads_per_block <= 0:
           raise ValueError("num_elems and threads_per_block must be positive")
       if num_elems % threads_per_block:
           raise ValueError("num_elems must be divisible by threads_per_block")
       size = num_elems // threads_per_block

       @cuda.jit
       def _kernel(src, dst):
           items = cuda.local.array(size, dtype=dtype)
           offset = cuda.threadIdx.x * size
           for i in range(size):
               items[i] = src[offset + i]
           for i in range(size):
               dst[offset + i] = items[i]

       return _kernel

   kernel1 = make_kernel1(h_src.size, threads_per_block, dtype)
   kernel1[1, threads_per_block](d_src, d_dst)
   d_dst.copy_to_host(h_dst)
   np.testing.assert_array_equal(h_dst, h_src)

This two-phase pattern makes a kernel before launching it. With
``coop.ThreadData``, one kernel can take the per-thread item count as an
argument and let cooperative operations supply the dtype:

.. code-block:: python

   from cuda import coop

   @cuda.jit
   def kernel(src, dst, items_per_thread):
       items = coop.ThreadData(items_per_thread)
       block = coop.this_block()
       coop.load(block, src, items)
       coop.store(block, dst, items)

   # The same kernel handles either dtype and either full-block input size.
   for dtype in (np.int32, np.float64):
       for items_per_thread in (1, 4):
           h_src = np.arange(threads_per_block * items_per_thread, dtype=dtype)
           d_src = cuda.to_device(h_src)
           d_dst = cuda.device_array_like(d_src)
           kernel[1, threads_per_block](d_src, d_dst, items_per_thread)
           np.testing.assert_array_equal(d_dst.copy_to_host(), h_src)

The compiler handles the payload's dtype and extent as follows:

* Load infers the dtype of ``items`` from ``src``. Store checks that
  ``dst.dtype`` agrees with it and reports a mismatch before kernel launch.
  No explicit payload dtype is needed in this kernel.
* ``items_per_thread`` supplies the element count for each thread. Storage
  size follows from that count and the inferred dtype. ``cuda.local.array``
  also computes storage bytes from its shape and dtype; ``ThreadData`` adds
  inference and the common payload interface. The count must still be a
  positive compile-time integer. Numba-CUDA-MLIR specializes the kernel for
  the argument's value; ``ThreadData`` does not create dynamically sized
  local arrays.

Inference avoids a separately written payload dtype that could become stale
when the input type changes. It does not check that the input and output
arrays have enough elements for the requested tile; these examples provide
one full tile per launch.

The ``items_per_thread`` field corresponds to the template parameter
``ITEMS_PER_THREAD`` or ``ItemsPerThread`` on the CUB C++ side, which is used
by many CUB block and warp primitives. A raw ``cuda.local.array`` does not
expose the common :class:`~cuda.coop.ThreadDataLike` interface; interpreting
its shape as an item count requires compiler-specific support.

Primitives use the field when generating the C++ wrapper compiled by NVRTC.
The compiler also knows which operands can supply the payload dtype. For Load
and Store, these are the source and destination arrays; the compiler checks
that their types agree with the same ``ThreadData`` payload. Other producers
follow their documented type-inference rules.

.. _coop-faq-local-array-payload:

Can I still use ``cuda.local.array`` instead of ``coop.ThreadData`` if I want?
--------------------------------------------------------------------------------

For per-thread array payloads in the common ``cuda.coop`` API, use
``coop.ThreadData`` (or a backend-supported implementation of
:class:`~cuda.coop.ThreadDataLike`; see :ref:`per-thread payloads
<coop-thread-data>`). A raw ``cuda.local.array`` does not implement that
common interface. Passing one to common ``coop.load`` or ``coop.store``
is rejected during kernel compilation. Static type checkers can also
report incompatible arguments when their types are available; an IDE or
linter does not necessarily perform the compiler's checks.

The Numba-qualified API, ``cuda.coop.numba_mlir``, does support fixed-size,
one-dimensional local-array payloads for operations that accept per-thread
arrays. That integration can obtain the item count from the array's shape
and the dtype from its Numba type. See the :ref:`qualified API comparison
<coop-programming-api-choice>` for an example. This support is specific to
Numba-CUDA-MLIR; ``coop.ThreadData`` supplies the shared payload contract
across compiler backends and the inference described above.

.. _coop-faq-thread-data-dtype:

When does ``ThreadData`` need an explicit element type?
-------------------------------------------------------

Start with an inferred payload:

.. code-block:: python

   # Inside a kernel; source and items_per_thread are kernel arguments.
   items = coop.ThreadData(items_per_thread)
   coop.load(coop.this_block(), source, items)

Load supplies the source element type. In Numba-CUDA-MLIR, supported indexed
assignments and a Store destination can also establish the type. Other
cooperative producers define their output types where their contracts say so.
Inference follows the backend's supported operations and assignments.

Numba's current planner cannot infer a payload type solely from the results of
a non-inlined device helper when no other supported operation supplies type
context. Cast the assigned result to the intended scalar type, such as
``numpy.int32``, or provide that context through a supported operation such as
Store. This is a limit of the current planner; the helper's result may already
have a type that the later compiler phases can determine.

For example, this helper returns ``int32``, but its call is deliberately
left out of line. The planner cannot use that return type to infer the
element type of ``items``:

.. code-block:: python

   import numpy as np
   from numba_cuda_mlir import cuda

   from cuda import coop

   @cuda.jit(device=True, inline="never")
   def make_value(index):
       return np.int32(index + 1)

   @cuda.jit
   def missing_context(dst, items_per_thread):
       items = coop.ThreadData(items_per_thread)
       offset = cuda.threadIdx.x * items_per_thread
       for i in range(items_per_thread):
           items[i] = make_value(offset + i)
       for i in range(items_per_thread):
           dst[offset + i] = items[i]

   items_per_thread = 4
   d_dst = cuda.device_array(32 * items_per_thread, dtype=np.int32)
   # Fails during compilation: the planner cannot infer items' dtype.
   missing_context[1, 32](d_dst, items_per_thread)

Although ``dst`` has a known element type, the ordinary indexed assignment
``dst[offset + i] = items[i]`` does not propagate that type back to
``items`` in this planner. Either of the following kernels supplies the
missing context:

.. code-block:: python

   @cuda.jit
   def typed_assignment(dst, items_per_thread):
       items = coop.ThreadData(items_per_thread)
       offset = cuda.threadIdx.x * items_per_thread
       for i in range(items_per_thread):
           # The cast is visible to the planner in the calling kernel.
           items[i] = np.int32(make_value(offset + i))
       for i in range(items_per_thread):
           dst[offset + i] = items[i]

   @cuda.jit
   def store_context(dst, items_per_thread):
       items = coop.ThreadData(items_per_thread)
       offset = cuda.threadIdx.x * items_per_thread
       for i in range(items_per_thread):
           items[i] = make_value(offset + i)
       # Cooperative Store supplies the dtype from dst.
       coop.store(coop.this_block(), dst, items)

   for items_per_thread in (1, 4):
       expected = np.arange(1, 32 * items_per_thread + 1, dtype=np.int32)
       for kernel in (typed_assignment, store_context):
           d_dst = cuda.device_array(expected.size, dtype=np.int32)
           kernel[1, 32](d_dst, items_per_thread)
           np.testing.assert_array_equal(d_dst.copy_to_host(), expected)

Both working kernels initialize every value in one block. The first gets
the payload dtype from the cast at the assignment; the second gets it from
Store's destination, even though Store appears after the assignments.
Neither needs an explicit ``dtype`` on ``ThreadData``.

An explicit signature on the scalar helper does not supply that missing
context to this early planner.

Helpers containing cooperative operations follow the separate
:ref:`device-helper inlining rules <coop-numba-device-helpers>`. Those rules
also cover passing groups and ``ThreadData`` payloads between helpers and
the calling kernel.

Use typed values when the computation needs a particular width or precision.
The optional ``dtype`` parameter supplies element-type information when the
surrounding program cannot establish it. It does not initialize the payload,
resolve conflicting typed values, or enable unsupported types. Initialize
every item before reading it.

Advanced CUTLASS interop can expose a raw integer IR value with a width but
no signedness. Preserve that information in a typed producer or supply
explicit element-type metadata. See :ref:`CUTLASS element-type inference
<coop-cutlass-dtype-inference>` for this case.

.. _coop-faq-exclusive-storage:
.. _why-use-sharing-exclusive-instead-of-omitting-storage:
.. _why-use-tempstorage-sharing-exclusive:

Why use ``TempStorage`` with ``sharing="exclusive"``?
-----------------------------------------------------

Passing ``coop.TempStorage(sharing="exclusive")`` as ``temp_storage`` gives
distinct call sites separate scratch slices. Omitting ``temp_storage`` lets
the compiler choose the scratch layout and insert reuse barriers. It may
reuse scratch across compatible calls; omission does not guarantee a
separate slice for each call site.

The explicit ``TempStorage`` descriptor retains its capacity, alignment, and
synchronization controls. Two calls with separate slices need no barrier
solely to reuse each other's scratch. This can save that synchronization when
``auto_sync=False``, at the cost of more shared memory. Barriers required by
the algorithm or by application data dependencies still apply.

Repeated execution of one call site, including a loop, reuses its slice.
Synchronize before that reuse or set ``auto_sync=True`` to request trailing
reuse barriers. ``sharing`` controls layout independently of ``auto_sync``.
The backend accounts for cooperative scratch with either explicit or omitted
storage; ``exclusive`` is a choice about which calls may share its bytes.

.. _coop-faq-temp-storage:

Why or when would I provide my own ``TempStorage``?
---------------------------------------------------

Usually, you can omit it. The compiler allocates the scratch required by
an operation and inserts a barrier to make reuse safe. Direct, striped, and
vectorized Load/Store need no shared scratch.

An explicit descriptor lets you choose which supported block operations
share an allocation. For example, a transpose Load and Store can
share scratch while the values stay in each thread's payload:

.. code-block:: python

   # Inside a kernel; source, destination, and items_per_thread are arguments.
   block = coop.this_block()
   items = coop.ThreadData(items_per_thread)
   scratch = coop.TempStorage(auto_sync=True)
   coop.load(
       block, source, items, algorithm="transpose", temp_storage=scratch
   )
   coop.store(
       block, destination, items, algorithm="transpose", temp_storage=scratch
   )

The compiler sizes and aligns the shared region for both calls. It inserts
a barrier after each use. Construct the descriptor inside the kernel, and
keep application values in ``ThreadData`` or your own arrays.

A descriptor also lets you request capacity or alignment, or choose separate
slices with ``sharing="exclusive"``. Explicit descriptors default to
``auto_sync=False``, so the kernel must provide reuse barriers, including across
loop iterations. The example requests ``auto_sync=True`` to insert those
barriers automatically. See :ref:`exclusive scratch slices
<coop-faq-exclusive-storage>` for the tradeoff between memory and reuse
synchronization.

Both backends use explicit descriptors for block transpose-family Load/Store,
Block Scan, Block Merge Sort, Block Radix Sort, and TopK. Storage-free block
Load/Store accept a descriptor but do not use it. Warp operations manage
their own resources and reject explicit descriptors.
See the :ref:`shared storage model <coop-common-storage>`,
:ref:`Numba storage rules <coop-temp-storage>`, and the
:ref:`CUTLASS storage rules <coop-cutlass-storage>` for each family's limits.
Numba's restrictions on combining cooperative backing with user static or
dynamic shared arrays are specific to that backend.
Both backends also accept explicit block scratch for Adjacent Difference,
Discontinuity, Histogram, and both Run Length Decode forms. Batched Warp
Reduction manages its own resources and accepts no explicit descriptor.

.. _coop-faq-installed-extra:

Can ``cuda.coop`` tell which extra I installed?
-----------------------------------------------

Not reliably. Package metadata lists the extras a distribution offers and
the dependencies associated with them. It does not provide a portable record
of which extra was requested when installing. The same dependencies may also
have been installed separately or by another package.

``pip install cuda-coop`` and
``pip install "cuda-coop[numba-cuda-mlir-cu13]"`` install the same wheel,
including ``cuda.coop.numba_mlir`` and ``cuda.coop.cutlass``.
The base install declares no Python package dependencies. The extra only
adds the requirements in ``pyproject.toml`` that install the supported
Numba-CUDA-MLIR stack for CUDA 13. If those dependencies are already present
at supported versions, either command gives you the same usable API.
Registration selects which compiler hooks to activate in the running process.

Use ``coop.register("numba-cuda-mlir")`` to state that intent explicitly.
It works regardless of import order, is safe to repeat, and also accepts
``"numba_cuda_mlir"``. The backend dependencies must already be installed.
For CUTLASS, use ``coop.register("cutlass")`` with a runtime meeting the
:doc:`CUTLASS requirements <../coop_cutlass>`; an installation extra is not
yet available. See :ref:`backend registration <coop-backend-registration>`.

.. _coop-faq-topk-order:

Does TopK return sorted results?
--------------------------------

No. It selects the smallest or largest keys and places them in a blocked
output prefix without promising their order. Only the first
``min(k, valid_items)`` positions are defined. When keys tie at the selection
boundary, any of the tied keys may fill the remaining positions. Pair variants
keep each selected key attached to its value. Use a sorting primitive when
you need ordered output. See the :ref:`Numba <coop-topk>` and
:ref:`CUTLASS <coop-cutlass-topk>` TopK examples.

.. _coop-faq-global-sort:

Does sorting each block sort the whole array?
---------------------------------------------

Each call sorts only the selected group's tile. Several blocks therefore
produce independently sorted tiles. A globally sorted array requires an
algorithm that combines those tiles. Warp and logical-warp Merge Sort
likewise sort each participating group's tile independently. The
:doc:`Merge Sort <visualizations/merge-sort>` and
:doc:`radix sorting <visualizations/radix>` visualizations illustrate these
operations.

.. _coop-faq-neighbor-operations:

When should I use Adjacent Difference or Discontinuity?
-------------------------------------------------------

Use :func:`cuda.coop.adjacent_difference` to compute a value from each item
and its neighbor, such as the delta between successive samples. Use
:func:`cuda.coop.discontinuity` to produce flags that mark boundaries, such
as the first and last item of each equal-value run. It returns ``int32``
flags; ``mode="heads_and_tails"`` returns both payloads.

Tile boundaries matter for both operations. Supply a predecessor or
successor when comparisons must continue across tiles. A multiblock
kernel that reads global neighbors must preserve those source values
until all readers finish. The
:doc:`Adjacent Difference <visualizations/adjacent-difference>` and
:doc:`Discontinuity <visualizations/discontinuity>` examples use separate
input and output arrays.

.. _coop-faq-histogram-padding:

Can I zero-pad a partial Histogram tile?
----------------------------------------

Every input sample contributes to a bin, including a padded zero. A
zero-padded load therefore adds extra counts to bin zero. Histogram has
no ``valid_items`` parameter. Process complete tiles or handle the tail
separately with a kernel that counts only valid samples.

Output padding is different: returned counter slots whose bin index is
at least ``bins`` contain zero. Store the first ``bins`` counters using
the striped layout. See the
:doc:`Histogram explorer <visualizations/histogram>`.

.. _coop-faq-histogram-accumulation:

Does Histogram retain counters between calls?
---------------------------------------------

Each call returns fresh counts and preserves its samples. For repeated
tiles within a kernel, keep an accumulator payload and add the returned
counts to it. Each thread retains the same striped bin ownership when
the configuration stays fixed. The :doc:`tested accumulation example
<visualizations/histogram>` shows this pattern.

CUB's ``BlockHistogram::Composite`` accumulates into a caller's counter
buffer. That persistent state belongs to the buffer; it does not require
a persistent C++ Histogram object. The Python API exposes the fresh-count
operation, so neither a parent object nor retained counters in
``TempStorage`` are needed.

.. _coop-faq-rld-lifecycle:

Why do windowed and bulk Run Length Decode use different calls?
---------------------------------------------------------------

:func:`cuda.coop.run_length_decode` returns a fixed-size payload for the
window beginning at ``decoded_window_offset``. Use it when the kernel
needs to work with that window's values. It prepares the run table on
each call, even when calls share a ``TempStorage`` descriptor.

:func:`cuda.coop.run_length_decode_into` writes the entire expanded
sequence into a destination array. It prepares the table once, loops
over decoding windows internally, and returns the total decoded size to
every thread. The destination must have enough remaining capacity;
use separate source and destination storage.

The prepared run table persists only within the call. A window offset
selects a position in the expanded sequence; the decoder does not advance
a hidden cursor. A shared scratch descriptor controls allocation reuse,
not decoder state. See :ref:`run positions and windows
<coop-glossary-decoding>` and the
:doc:`RLD explorer <visualizations/run-length-decode>`.

.. _coop-faq-rld-padding:

How do I pad run inputs and recognize the end of decoded output?
----------------------------------------------------------------

Use a positive prefix of run lengths followed by zeros. An all-zero tile
represents an empty sequence; an interior zero followed by a positive
length is invalid. The values associated with padding runs are ignored.

A windowed decode fills positions beyond the expanded sequence with
zero. Zero may also be a real run value, so use the total decoded size
to determine which positions are valid. The Numba-qualified API can write
that total and relative run offsets to auxiliary payloads; invalid
relative offsets contain the maximum value of the selected unsigned
offset dtype. Bulk decoding writes only valid items, leaving the rest
of the destination unchanged.

.. _coop-faq-batched-reduce:

How does Batched Warp Reduction differ from ordinary Reduce?
------------------------------------------------------------

Ordinary ``reduce(group, values)`` combines all participating threads'
payload items into one aggregate. Both block and Warp reductions accept
one scalar or a fixed-size payload per thread and define the result only
at group rank zero.
``reduce_batched(warp, values)`` reduces each local slot independently across
the warp. Three slots per lane mean three independent results, one for each
slot.

The results are distributed among lanes in blocked or striped order;
they are not broadcast to every lane. Each returned payload has
``ceil(batches / warp_width)`` slots, and slots without a corresponding
batch are unspecified. The :doc:`feature-sum example
<visualizations/reduce-batched>` guards its stores by batch index.

.. _coop-faq-compiler-setup:

Why does import work but kernel compilation fail?
-------------------------------------------------

The common namespace can be imported without a compiler or GPU. Compilation
also needs a compatible backend runtime, its compiler hooks, and the CUDA
toolkit selected by that runtime. Call ``coop.register("numba-cuda-mlir")``
or ``coop.register("cutlass")`` explicitly before compiling to report a
missing or incompatible backend at setup time. The switch
``CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION`` disables only automatic probing;
explicit registration and qualified imports still work.

Check the operation's launch shape, dtype, and participation requirements.
Both backends implement the common contract; qualified extensions follow
their compiler's guide. In a process using both DSLs, keep Numba values
inside Numba kernels and CuTe values inside CuTe kernels.

For provider compilation errors, verify the toolkit and matching CCCL
headers, then use ``CUDA_COOP_SOURCE_DUMP_DIR`` to inspect generated source.
The :doc:`Numba-CUDA-MLIR Developer Guide <developer_overview>` and
:doc:`CUTLASS Developer Guide <cutlass_developer_guide>` explain their
compiler and linker diagnostics. CUTLASS's
:ref:`runtime requirements <coop-cutlass-requirements>` remain a separate
prerequisite; a successful host import does not qualify a public runtime.
