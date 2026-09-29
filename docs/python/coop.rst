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
   coop/configuration

``cuda.coop`` brings CCCL's optimized cooperative algorithms to Python GPU
kernels. Use it when threads need to work together, such as loading a tile
of values or arranging data for the next stage of a computation. These
operations run inside a kernel, where you can combine them with your own
code and reuse algorithms maintained and tuned for NVIDIA GPUs.

The shared API describes cooperative Load and Store operations using CUB.
It supplies the planning and headers that compiler integrations need to
lower those calls to device code. This package layer contains the shared
core; executing a kernel also requires a compiler integration.

Get started
-----------

Install the dependency-free base package:

.. code-block:: console

   python -m pip install cuda-coop

The package provides the shared API and matching CCCL headers. Importing
it does not require a CUDA device. See :doc:`configuration
<coop/configuration>` for package and header details.

Inside a kernel
---------------

A cooperative operation acts on a *group* of threads, usually a warp or a
block. Each thread owns part of the group's data, held in ``ThreadData``.
Together, those parts form a tile.

The :doc:`programming concepts <coop/concepts>` explain participation,
data layouts, operation results, and temporary storage.

Common API and compiler extensions
----------------------------------

Use ``from cuda import coop`` for the common API. It describes the
cooperative part of a kernel independently of its compiler.

Explore further
---------------

Use the :doc:`API reference <coop_api>` for signatures and return values,
and :doc:`configuration <coop/configuration>` for package and compiler
settings.

.. raw:: html

   <span id="backend-registration"></span>
   <span id="build-time-cmake-variables"></span>
   <span id="compilation-and-headers"></span>
   <span id="configuration"></span>
   <span id="coop-backend-registration"></span>
   <span id="coop-data-layouts"></span>
   <span id="coop-participation"></span>
   <span id="coop-temp-storage"></span>
   <span id="coop-thread-data"></span>
   <span id="coop-thread-groups"></span>
   <span id="data-layouts-and-algorithms"></span>
   <span id="groups-and-thread-data"></span>
   <span id="installation"></span>
   <span id="kernel-api"></span>
   <span id="load-and-store-semantics"></span>
   <span id="participation-and-synchronization"></span>
   <span id="per-thread-payloads"></span>
   <span id="runtime-environment-variables"></span>
   <span id="temporary-storage"></span>
