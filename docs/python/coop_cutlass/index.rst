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

The distribution is ``cuda-coop-cutlass``. It contributes the
``cuda.coop.cutlass`` package to the shared ``cuda`` namespace; ``cuda.coop``
is its namespace parent.

Installation
------------

The standalone package requires CUDA Toolkit 13 and CUTLASS CuTe DSL
4.9.0.dev1 or later in the 4.x series. Use a GPU architecture supported by
your CuTe compiler. The earliest target in 4.9.0.dev1 is compute capability
8.0 (SM80). One tested Linux configuration pairs an NVIDIA L4 with CUDA
Toolkit 13.4 and CuTe DSL 4.9.0.dev1.

Install from a checkout containing this project:

.. code-block:: bash

   python -m pip install ./python/cuda_coop_cutlass

Use a driver compatible with your toolkit. The toolkit's NVRTC compiles
the CUB device code to LTO-IR. CuTe's final linker must be at least as
new as NVRTC within the same CUDA major series. The Linux CUDA 13 wheel
for CuTe DSL 4.9.0.dev1 uses a CUDA 13.4 linker. CUDA Toolkits 13.3 and
13.4 have been tested with this wheel. CUDA 13.5 LTO-IR needs a newer
linker. See
`nvJitLink compatibility <https://docs.nvidia.com/cuda/nvjitlink/#compatibility>`__.

The wheel includes the required CCCL headers and licenses. The
``nvidia-cutlass-dsl[cu13]`` dependency selects CuTe's CUDA 13 libraries.
The :ref:`compiler requirements <coop-cutlass-compiler-requirements>`
describe the external LTO-IR linking and trace-finalization interfaces.

Start with the :doc:`Programming Guide <programming_guide>` for a complete
prefix-sum kernel and :doc:`concepts` for participation, layouts, ownership,
and storage. The :doc:`examples` page shows executable programs with CPU
reference checks and downloads. The :doc:`API reference <api>` lists every operation and
its controls. Developers new to CUTLASS can read :doc:`cutlass_primer`
before the :doc:`Developer Guide <developer_guide>`.

.. toctree::
   :maxdepth: 2

   concepts
   programming_guide
   api
   examples
   tensor_memory_topk
   developer_guide
   cutlass_primer
   cutlass_integration_example
   configuration
   faqs
   glossary
   visualizations/index
