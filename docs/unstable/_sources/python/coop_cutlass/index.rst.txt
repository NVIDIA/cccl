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

The standalone package requires CUTLASS DSL 4.9.0.dev1 or later in the 4.x
series and CUDA 13 on Linux. Install from a checkout containing this project:

.. code-block:: bash

   python -m pip install ./python/cuda_coop_cutlass

Use a compatible CUDA driver and toolkit. NVRTC compiles the CUB providers,
and the CuTe compiler must support external NVIDIA LTO-IR linking and
trace-finalization hooks. The wheel includes the required CCCL headers and
licenses. Package metadata selects the CuTe CUDA 13 libraries through the
``nvidia-cutlass-dsl[cu13]`` dependency, including the 4.9.0.dev1 prerelease.

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
