.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-overview:

``cuda.coop.cutlass``: Cooperative Primitives for CuTe Kernels
==============================================================

``cuda.coop.cutlass`` brings CUB's block and warp primitives to kernels
written in the CUTLASS CuTe DSL. The threads of a block or warp call one
function together to load and store data, reduce, scan, sort, select and more.

This kernel computes a running sum across a block of 128 threads:

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/prefix_sum.py
   :language: python
   :start-after: docs: start cutlass-prefix-sum
   :end-before: docs: end cutlass-prefix-sum

The :doc:`Programming Guide <programming_guide>` shows the whole program and
explains each line.

.. _coop-cutlass-installation:

Installation
------------

From the repository root, install the package:

.. code-block:: bash

   python -m pip install ./python/cuda_coop_cutlass

.. _coop-cutlass-requirements:

Requirements
~~~~~~~~~~~~

* Python 3.10 or later.
* Linux with CUDA 13, an NVIDIA GPU and its driver.
* CUTLASS CuTe DSL 4.9.0.dev1 or later in the 4.x series. The package depends
  on ``nvidia-cutlass-dsl[cu13]``, which brings CuTe's CUDA 13 libraries.

The package is tested with CuTe in its default mode, which leaves PyIR off.

The wheel includes the CCCL headers it needs. The CUB code behind each
primitive is compiled with NVRTC when a kernel is compiled, and linked into
that kernel. :ref:`Compiler requirements <coop-cutlass-compiler-requirements>`
lists the CuTe interfaces this depends on.

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
