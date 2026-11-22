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
* Linux with CUDA Toolkit 13 and a compatible driver.
* A GPU architecture supported by your CuTe compiler. The earliest target
  in CuTe DSL 4.9.0.dev1 is compute capability 8.0 (SM80).
* CUTLASS CuTe DSL 4.9.0.dev1 or later in the 4.x series. The package depends
  on ``nvidia-cutlass-dsl[cu13]``, which brings CuTe's CUDA 13 libraries.

One tested combination is an NVIDIA L4, CUDA Toolkit 13.4 and CuTe DSL
4.9.0.dev1 with PyIR off, which is CuTe's default.

The wheel includes the CCCL headers it needs. The toolkit's NVRTC compiles
the CUB device code to LTO-IR. To link it into the kernel, CuTe needs
an nvJitLink version at least as new as NVRTC in the same CUDA major
series. For example, a CUDA 13.4 linker accepts CUDA 13.3 or 13.4 LTO-IR.
CUDA 13.5 LTO-IR requires a CUDA 13.5 or later linker in the 13.x series. See
`nvJitLink compatibility <https://docs.nvidia.com/cuda/nvjitlink/#compatibility>`__
and :ref:`Compiler requirements <coop-cutlass-compiler-requirements>`.

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
