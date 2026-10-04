.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _cccl-python-coop:

``cuda.coop``: Cooperative Group Primitives
===========================================

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Shared concepts

   Overview <self>
   coop/concepts
   coop/visualizations/index
   coop/glossary
   coop/faqs
   coop/configuration

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Numba-CUDA-MLIR

   coop/programming_guide
   coop/developer_overview
   coop/debugger_walkthrough_temp_storage

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: CUTLASS

   coop_cutlass
   coop/cutlass_developer_guide

``cuda.coop`` brings CCCL's optimized cooperative algorithms to Python GPU
kernels. Use it when threads need to work together, such as summing a tile
of values or arranging data for the next stage of a computation. These
operations run inside a kernel, where you can combine them with your own
code and reuse algorithms maintained and tuned for NVIDIA GPUs.

The common Python API works with Numba-CUDA-MLIR and CUTLASS CuTe DSL, using
CUB underneath. You keep your compiler's kernel syntax and launch
conventions, and use ``cuda.coop`` for the cooperative parts of the work.

.. raw:: html

   <span id="installation"></span>

Get started
-----------

Choose the guide for the kernel language you use:

* The :doc:`Numba-CUDA-MLIR Programming Guide <coop/programming_guide>`
  starts with a complete kernel and host code to launch it.
* The :doc:`CUTLASS Programming Guide <coop_cutlass>` shows how to use the
  same operations in CuTe kernels and work with CuTe register tensors.

For Numba-CUDA-MLIR, install the extra matching your CUDA major version:

.. code-block:: console

   python -m pip install "cuda-coop[numba-cuda-mlir-cu13]"
   # Use numba-cuda-mlir-cu12 with CUDA 12.

.. raw:: html

   <span id="coop-numba-validation"></span><span id="numba-cuda-mlir-validation-scope"></span>
   <span id="coop-numba-context-lifetime"></span><span id="cuda-devices-and-context-lifetime"></span>

Check the Numba guide's :ref:`requirements and kernel reuse limits
<coop-numba-requirements>` before running your first kernel.

For CUTLASS, install the base package alongside a compatible CuTe runtime:

.. code-block:: console

   python -m pip install cuda-coop

Check the :ref:`CUTLASS runtime requirements <coop-cutlass-requirements>`
before installing: this integration currently requires a compatible
Linux/CUDA 13 environment, and an official public runtime artifact has yet
to be qualified.

Inside a kernel
---------------

A cooperative operation acts on a *group* of threads, usually a warp or a
block. Each thread owns part of the group's data, held in a ``ThreadData``
object. Together, those parts form a tile.

For example, these calls load a full tile, compute its prefix sums, and
write the result back to memory:

.. code-block:: python

   # Inside a kernel, with `coop` imported from `cuda`:
   block = coop.this_block()
   items = coop.ThreadData(items_per_thread)

   coop.load(block, source, items, offset=offset)
   prefixes = coop.exclusive_sum(block, items)
   coop.store(block, destination, prefixes, offset=offset)

The kernel supplies the tile's ``offset`` and a compile-time
``items_per_thread`` count. With 128 threads and four items per thread, the
block processes 512 values. Each block computes its own prefix sum; a
device-wide scan also needs to combine results across blocks.

All threads in the group must reach these calls. For a final, partial tile,
use the operations' valid-item controls while keeping the whole group
involved. Both programming guides show the complete code, including tail
handling and the compiler's launch syntax.

.. raw:: html

   <span id="shared-execution-model"></span><span id="groups-and-tiles"></span>
   <span id="participation-and-valid-prefixes"></span><span id="per-thread-payloads"></span>
   <span id="layouts-and-operation-order"></span><span id="result-ownership"></span>
   <span id="scratch-allocation-and-reuse"></span>
   <span id="coop-common-groups"></span><span id="groups-and-thread-data"></span>
   <span id="coop-common-participation"></span><span id="load-and-store-semantics"></span>
   <span id="coop-common-payloads"></span><span id="coop-common-layouts"></span>
   <span id="exchange-semantics"></span><span id="shuffle-semantics"></span>
   <span id="scan-semantics"></span><span id="coop-common-results"></span>
   <span id="coop-common-storage"></span><span id="temporary-storage"></span>

The :doc:`programming concepts <coop/concepts>` explain participation,
data layouts, result ownership, and temporary storage in detail.

.. _coop-backends:

Common API and compiler extensions
----------------------------------

Use ``from cuda import coop`` for the common API. Numba-CUDA-MLIR and
CUTLASS implement its operations with the same argument conventions and
behavior. The surrounding kernel code, including array arguments and
launches, follows your compiler's conventions.

The qualified namespaces, ``cuda.coop.numba_mlir`` and
``cuda.coop.cutlass``, include the common operations and add features for
their compiler. For example, the Numba API accepts device callbacks, and
the CUTLASS API can convert existing CuTe register tensors to ``ThreadData``.
The :ref:`Numba <coop-programming-api-choice>` and
:ref:`CUTLASS <coop-cutlass-api-choice>` guides explain when to use these
extensions.

.. raw:: html

   <span id="backend-coverage"></span><span id="common-and-qualified-apis"></span>
   <span id="calling-conventions"></span><span id="registering-a-backend"></span>
   <span id="block-prefix-callbacks"></span>
   <span id="coop-common-api"></span><span id="coop-api-namespaces"></span>
   <span id="kernel-api"></span><span id="portable-and-qualified-apis"></span>
   <span id="coop-common-calling-conventions"></span>
   <span id="coop-backend-registration"></span>

Import your kernel compiler before ``cuda.coop`` to activate its integration
automatically. In a notebook or library where import order can vary, use
explicit :ref:`backend registration <coop-backend-registration>`.

Explore further
---------------

The :doc:`interactive visualizations <coop/visualizations/index>` let you
step through operations and see which values each thread owns. Use the
:doc:`API reference <coop_api>` to find an operation's parameters and
return values, or the :doc:`FAQs <coop/faqs>` for practical questions.

.. raw:: html

   <span id="configuration"></span><span id="runtime-environment-variables"></span>
   <span id="build-time-cmake-variables"></span><span id="compilation-and-headers"></span>

For cache and compiler settings, see :doc:`configuration <coop/configuration>`.
To work on the integration itself, follow a kernel through the
:doc:`Numba-CUDA-MLIR <coop/developer_overview>` or
:doc:`CUTLASS <coop/cutlass_developer_guide>` developer guide.
