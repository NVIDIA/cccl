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

The standalone package targets the CUTLASS 4.9 CuTe compiler integration and
CUDA 13 on Linux. Install from a checkout containing this project:

.. code-block:: bash

   python -m pip install ./python/cuda_coop_cutlass

Use a compatible CUDA driver and toolkit. NVRTC compiles the CUB providers,
and the CuTe compiler must support external NVIDIA LTO-IR linking and
trace-finalization hooks. The wheel includes the required CCCL headers and
licenses. Runtime prerequisites are declared by the package metadata.

This draft has been developed with a retained CUTLASS 4.9 candidate. Candidate
validation and released-wheel installation are separate qualification gates;
see the draft PR's validation record for the exact compiler and toolkit used.
A public package release and package uploads are not part of this draft.

Start with the :doc:`Programming Guide <programming_guide>` for a complete
prefix-sum kernel and :doc:`concepts` for participation, layouts, ownership,
and storage. The :doc:`examples` page links executable programs with CPU
reference checks. The :doc:`API reference <api>` lists every operation and
its controls.

.. toctree::
   :maxdepth: 2

   concepts
   programming_guide
   api
   examples
   developer_guide
   configuration
   faqs
   glossary
   neighbor-operations
   visualizations/index
