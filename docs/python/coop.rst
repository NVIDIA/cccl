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

``cuda.coop`` brings CCCL's optimized cooperative algorithms to Python GPU
kernels. Use it when threads need to work together, such as summing a tile
of values or arranging data for the next stage of a computation. These
operations run inside a kernel, where you can combine them with your own
code and reuse algorithms maintained and tuned for NVIDIA GPUs.

The Numba-CUDA-MLIR integration compiles cooperative calls inside Python
GPU kernels. You keep Numba's kernel syntax and launch conventions and
use ``cuda.coop`` for the cooperative parts of the work.

Get started
-----------

Start with the :doc:`Numba-CUDA-MLIR Programming Guide
<coop/programming_guide>` for complete kernels and host launch code.
Install the extra matching your CUDA major version:

.. code-block:: console

   python -m pip install "cuda-coop[numba-cuda-mlir-cu13]"
   # Use numba-cuda-mlir-cu12 with CUDA 12.

The guide covers :ref:`requirements and kernel reuse
<coop-numba-requirements>`.

Inside a kernel
---------------

A cooperative operation acts on a *group* of threads, usually a warp or a
block. Each thread owns part of the group's data, held in ``ThreadData``.
Together, those parts form a tile.

These calls load a full tile, compute its prefix sums, and write the
result back to memory:

.. code-block:: python

   # Inside a kernel, with `coop` imported from `cuda`:
   block = coop.this_block()
   items = coop.ThreadData(items_per_thread)

   coop.load(block, source, items, offset=offset)
   prefixes = coop.exclusive_sum(block, items)
   coop.store(block, destination, prefixes, offset=offset)

The kernel supplies the tile's ``offset`` and a compile-time
``items_per_thread`` count. With 128 threads and four items per thread,
the block processes 512 values. All threads in the group must reach the
cooperative calls, including when only part of the tile is valid.

Each block computes its own prefix sum; a device-wide scan also needs
to combine results across blocks.

The :doc:`programming concepts <coop/concepts>` explain participation,
data layouts, operation results, and temporary storage.

Common API and compiler extensions
----------------------------------

Use ``from cuda import coop`` for the common API. It describes the
cooperative part of a kernel independently of its compiler.

The qualified ``cuda.coop.numba_mlir`` namespace provides Numba-specific
operands and controls. The :ref:`Numba API comparison
<coop-programming-api-choice>` explains when to use it.

Import your kernel compiler before ``cuda.coop`` to activate its
integration automatically. For varying import order, use explicit
:ref:`backend registration <coop-backend-registration>`.

Explore further
---------------

The :doc:`interactive visualizations <coop/visualizations/index>` show
which values each thread owns and how an operation moves them.

Use the :doc:`API reference <coop_api>` for signatures and return values,
and :doc:`configuration <coop/configuration>` for package and compiler
settings.

To work on the integration, follow a kernel through the
:doc:`Numba-CUDA-MLIR Developer Guide <coop/developer_overview>`.

.. raw:: html

   <span id="block-prefix-callbacks"></span>
   <span id="build-time-cmake-variables"></span>
   <span id="compilation-and-headers"></span>
   <span id="configuration"></span>
   <span id="coop-backend-registration"></span>
   <span id="coop-numba-context-lifetime"></span>
   <span id="coop-numba-validation"></span>
   <span id="cuda-devices-and-context-lifetime"></span>
   <span id="exchange-semantics"></span>
   <span id="groups-and-thread-data"></span>
   <span id="installation"></span>
   <span id="kernel-api"></span>
   <span id="load-and-store-semantics"></span>
   <span id="numba-cuda-mlir-validation-scope"></span>
   <span id="registering-a-backend"></span>
   <span id="runtime-environment-variables"></span>
   <span id="scan-semantics"></span>
   <span id="shuffle-semantics"></span>
   <span id="temporary-storage"></span>
