.. _cudax-memory-resource:

Memory Resources
=================

.. toctree::
   :glob:
   :maxdepth: 3

   ${repo_docs_api_path}/enum*cuda__memory__pool*
   ${repo_docs_api_path}/struct*cuda__memory__pool__properties*
   ${repo_docs_api_path}/class*cuda__memory__pool*

The ``<cuda/experimental/memory_resource>`` header provides a standard C++ interface for *heterogeneous*, *stream-ordered* memory
allocation tailored to the needs of CUDA C++ developers. This design builds off of the success of the `RAPIDS Memory Manager (RMM) <https://github.com/rapidsai/rmm>`__
project and evolves the design based on lessons learned.

``<cuda/experimental/memory_resource>`` is not intended to replace RMM, but instead moves the definition of the memory allocation
interface to a more centralized home in CCCL. RMM will remain as a collection of implementations of the ``cuda::mr``
interfaces.
