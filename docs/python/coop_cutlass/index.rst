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

On Linux with a CUDA 13-capable NVIDIA driver, install the published package
with its CUDA Toolkit 13.4 headers, NVRTC and nvJitLink:

.. code-block:: bash

   python -m pip install 'cuda-coop-cutlass[cu13]'

To build from source, run the following from the CCCL repository root:

.. code-block:: bash

   python -m pip install './python/cuda_coop_cutlass[cu13]'

With a compatible system toolkit already installed, omit the ``[cu13]`` extra
from either command.

.. _coop-cutlass-requirements:

Requirements
~~~~~~~~~~~~

* Python 3.10 or later.
* CUDA Toolkit 13 and a compatible driver. On Linux with CuTe DSL
  4.9.0.dev1, use NVRTC 13.1 through 13.4.
* On Windows, linking kernels requires nvJitLink 13.4 or later in the
  CUDA 13 series. The ``[cu13]`` extra installs a compatible linker.
* A GPU architecture supported by your CuTe compiler. The earliest target
  in CuTe DSL 4.9.0.dev1 is compute capability 8.0 (SM80).
* CUTLASS CuTe DSL 4.9.0.dev1 or later in the 4.x series. The package depends
  on ``nvidia-cutlass-dsl[cu13]``, which brings CuTe's CUDA 13 libraries.

One tested Linux configuration pairs an NVIDIA L4 with CUDA Toolkit 13.4
and CuTe DSL 4.9.0.dev1. It uses CuTe's default setting of PyIR off.

The wheel includes the CCCL headers it needs. The toolkit's NVRTC compiles
the CUB device code to LTO-IR. On Linux, CuTe's final linker must be at
least as new as NVRTC within the same CUDA major series. The Linux CUDA 13
wheel for CuTe DSL 4.9.0.dev1 uses a CUDA 13.4 linker. CUDA Toolkits 13.3 and
13.4 have been tested with this wheel. CUDA 13.5 LTO-IR needs a newer
linker. See
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
