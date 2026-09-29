.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-faqs:

FAQs
====

.. _coop-faq-namespaces:

Why are there both ``cuda.coop`` and ``cuda.coop.numba_mlir``?
--------------------------------------------------------------

``cuda.coop`` provides the common API for cooperative operations. A kernel
compiler's backend implements those calls. Start with this namespace when
its groups, ``ThreadData`` payloads, and built-in operators cover your needs:

.. code-block:: python

   from cuda import coop

   coop.register("numba-cuda-mlir")

``cuda.coop.numba_mlir`` exposes that backend's API, including extensions
specific to Numba-CUDA-MLIR. Use it for features such as fixed-size Numba
local-array payloads:

.. code-block:: python

   import cuda.coop.numba_mlir as numba_coop

Importing this namespace also registers the backend. Both namespaces can
appear in one kernel, and shared operations follow the same contracts.
See the :ref:`API comparison <coop-programming-api-choice>` for the
operation-specific differences.

Numba-CUDA-MLIR is the first backend; CUTLASS support is planned. The common
API gives libraries a compiler-independent way to express cooperative
operations. Kernel launch syntax and other DSL-specific code still need
adaptation when moving to another compiler.

.. _coop-faq-numba-only:

I only use Numba-CUDA-MLIR. Can I import its namespace as ``coop``?
-------------------------------------------------------------------

Yes. This is supported and registers the backend:

.. code-block:: python

   from numba_cuda_mlir import cuda

   import cuda.coop.numba_mlir as coop

You can use common operations and backend extensions through that one name.
The documentation uses ``numba_coop`` when showing backend calls alongside
common calls, so readers can see which API an example needs.

Keep the alias on a dotted import. Bare ``import cuda.coop.numba_mlir``
assigns the top-level package to ``cuda`` in that scope, replacing the name
previously imported from ``numba_cuda_mlir``.

.. _coop-faq-thread-data-dtype:

When does ``ThreadData`` need an explicit element type?
-------------------------------------------------------

Start with an inferred payload:

.. code-block:: python

   # Inside a kernel; source is a typed memory operand.
   items = coop.ThreadData(items_per_thread=2)
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

Use typed values when the computation needs a particular width or precision.
The optional ``dtype`` parameter supplies element-type information when the
surrounding program cannot establish it. It does not initialize the payload,
resolve conflicting typed values, or enable unsupported types. Initialize
every item before reading it.

.. _coop-faq-exclusive-storage:

Why use ``sharing="exclusive"`` instead of omitting storage?
------------------------------------------------------------

Omitting ``temp_storage`` lets the compiler choose the scratch layout and
insert reuse barriers. It may reuse scratch across compatible calls; omission
does not guarantee a separate slice for each call site.

``TempStorage(sharing="exclusive")`` gives distinct call sites separate
slices while retaining the descriptor's capacity, alignment, and
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

   # Inside a kernel; source and destination are kernel arguments.
   block = coop.this_block()
   items = coop.ThreadData(items_per_thread=2)
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

The current backend accepts explicit descriptors for block transpose-family Load/Store.
Warp operations use compiler-owned scratch. See
:ref:`temporary storage <coop-temp-storage>` for the complete contract and
shared-memory restrictions.

.. _coop-faq-installed-extra:

Can ``cuda.coop`` tell which extra I installed?
-----------------------------------------------

Not reliably. Package metadata lists the extras a distribution offers and
the dependencies associated with them. It does not provide a portable record
of which extra was requested when installing. The same dependencies may also
have been installed separately or by another package.

``pip install cuda-coop`` and
``pip install "cuda-coop[numba-cuda-mlir-cu13]"`` install the same wheel,
including ``cuda.coop.numba_mlir`` and every other shipped DSL integration.
The base install declares no Python package dependencies. The extra only
adds the requirements in ``pyproject.toml`` that install the supported
Numba-CUDA-MLIR stack for CUDA 13. If those dependencies are already present
at supported versions, either command gives you the same usable API.
Registration selects which compiler hooks to activate in the running process.

Use ``coop.register("numba-cuda-mlir")`` to state that intent explicitly.
It works regardless of import order, is safe to repeat, and also accepts
``"numba_cuda_mlir"``. The backend dependencies must already be installed.
See :ref:`backend registration <coop-backend-registration>`.
