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
local-array payloads, device callbacks, or Scan prefix callbacks:

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

.. _coop-faq-temp-storage:

Why or when would I provide my own ``TempStorage``?
---------------------------------------------------

Usually, you can omit it. The compiler allocates the scratch required by
an operation and inserts a barrier to make reuse safe. Direct, striped, and
vectorized Load/Store need no shared scratch.

An explicit descriptor is useful when several supported block operations
can reuse the same allocation. For example, a transpose Load and Store can
share scratch while the values stay in each thread's payload:

.. code-block:: python

   # Inside a kernel; source and destination are kernel arguments.
   block = coop.this_block()
   items = coop.ThreadData(2)
   scratch = coop.TempStorage()
   coop.load(
       block, source, items, algorithm="transpose", temp_storage=scratch
   )
   coop.store(
       block, destination, items, algorithm="transpose", temp_storage=scratch
   )

The compiler sizes and aligns the shared region for both calls. It inserts
a barrier after each use. Construct the descriptor inside the kernel, and
keep application values in ``ThreadData`` or your own arrays.

A descriptor also lets you request capacity or alignment, choose separate
slices with ``sharing="exclusive"``, or take responsibility for reuse barriers
with ``auto_sync=False``. Keep automatic synchronization enabled unless your
kernel provides the required barriers itself, including across loop
iterations. Separate slices do not remove the need to protect reuse.

The current backend accepts explicit descriptors for block transpose-family
Load/Store and Block Scan. Warp Load/Store and Warp Scan use compiler-owned
storage and reject explicit descriptors. See
:ref:`temporary storage <coop-temp-storage>` for the complete contract and
shared-memory restrictions.

.. _coop-faq-installed-extra:

Can ``cuda.coop`` tell which extra I installed?
-----------------------------------------------

Not reliably. Package metadata lists the extras a distribution offers and
the dependencies associated with them. It does not provide a portable record
of which extra was requested when installing. The same dependencies may also
have been installed separately or by another package.

``pip install cuda-coop`` installs the common API with no Python package
dependencies. ``pip install "cuda-coop[numba-cuda-mlir-cu13]"`` also installs
the Numba-CUDA-MLIR backend's dependencies for CUDA 13. These commands select
what is available in the environment; registration selects which compiler
hooks to activate in the running process.

Use ``coop.register("numba-cuda-mlir")`` to state that intent explicitly.
It works regardless of import order, is safe to repeat, and also accepts
``"numba_cuda_mlir"``. The backend dependencies must already be installed.
See :ref:`backend registration <coop-backend-registration>`.
