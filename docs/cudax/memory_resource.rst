.. _cudax-memory-resource:

Memory Resources
=================

.. toctree::
   :glob:
   :maxdepth: 3

   ${repo_docs_api_path}/*any__resource*
   ${repo_docs_api_path}/enum*async__memory__pool*
   ${repo_docs_api_path}/struct*async__memory__pool__properties*
   ${repo_docs_api_path}/class*async__memory__pool*
   ${repo_docs_api_path}/class*async__memory__resource*

The ``<cuda/experimental/memory_resource.cuh>`` header provides:
   -  :ref:`any_resource <cudax-memory-resource-any-resource>` and
      :ref:`async_any_resource <cudax-memory-resource-async-any-resource>` type erased memory resources similar to
      ``std::any``. In contrast to :ref:`resource_ref <libcudacxx-extended-api-memory-resources-resource-ref>` they
      own the contained resource.
   -  :ref:`async_memory_resource <cudax-memory-resource-async>` A standard C++ interface for *heterogeneous*,
      *stream-ordered* memory allocation tailored to the needs of CUDA C++ developers. This design builds off of the
      success of the `RAPIDS Memory Manager (RMM) <https://github.com/rapidsai/rmm>`__ project and evolves the design
      based on lessons learned.

``<cuda/experimental/memory_resource.cuh>`` is not intended to replace RMM, but instead moves the definition of the
memory allocation interface to a more centralized home in CCCL. RMM will remain as a collection of implementations of
the ``cuda::mr`` interfaces.
