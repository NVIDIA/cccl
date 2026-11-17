.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-overview:

``cuda.coop.cutlass``
=====================

``cuda.coop.cutlass`` provides CUB cooperative primitives inside CuTe kernels.
Threads in a block or warp load and store tiles, reduce or scan values,
rearrange payloads, sort or select keys, compare neighbors, count samples,
and decode runs. CuTe compiles the enclosing kernel and links the generated
CUB device functions into it.

.. code-block:: python

   import cuda.coop.cutlass as coop

Installation
------------

From the repository root, install the standalone package:

.. code-block:: bash

   python -m pip install ./python/cuda_coop_cutlass

Requirements
~~~~~~~~~~~~

* Python 3.10 or later.
* CUDA 13 on Linux, with a compatible NVIDIA GPU and driver. NVRTC compiles
  the CUB device functions.
* CUTLASS CuTe DSL 4.9.0.dev1 or later in the 4.x series, with support for
  external NVIDIA LTO-IR linking and trace-finalization hooks. See
  :ref:`compiler requirements <coop-cutlass-compiler-requirements>` for the
  required interfaces.

The wheel includes the required CCCL headers and licenses. Package metadata
selects the CuTe CUDA 13 libraries through the ``nvidia-cutlass-dsl[cu13]``
dependency, including the 4.9.0.dev1 prerelease.

Documentation
-------------

.. toctree::
   :maxdepth: 2

   programming_guide
   examples
   concepts
   api
   advanced
   configuration
   faqs
   glossary
   visualizations/index
   developer_guide
