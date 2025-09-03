.. _cudax-memory-resource:

Memory Resources
=================

.. toctree::
   :glob:
   :maxdepth: 3

   api/*any__resource*
   api/struct*memory__pool__properties*
   api/class*device__memory__pool*
   api/class*device__memory__resource*
   api/class*pinned__memory__pool*
   api/class*pinned__memory__resource*
   api/*shared__resource*

The ``<cuda/experimental/memory_resource.cuh>`` header provides:
   -  :ref:`any_synchronous_resource <cudax-memory-resource-any-resource>` and
      :ref:`any_resource <cudax-memory-resource-any-async-resource>` type erased memory resources similar to
      ``std::any``. In contrast to :ref:`resource_ref <libcudacxx-extended-api-memory-resources-resource-ref>` they
      own the contained resource.
   -  :ref:`device_memory_resource <cudax-memory-resource-async>` A standard C++ interface for *heterogeneous*,
      *stream-ordered* memory allocation tailored to the needs of CUDA C++ developers. This design builds off of the
      success of the `RAPIDS Memory Manager (RMM) <https://github.com/rapidsai/rmm>`__ project and evolves the design
      based on lessons learned.
   -  :ref:`shared_resource <cudax-memory-resource-shared-resource>` a type erased reference counted memory resource.
      In contrast to :ref:`any_resource <cudax-memory-resource-any-resource>` it additionally provides shared ownership
      semantics.

``<cuda/experimental/memory_resource.cuh>`` is not intended to replace RMM, but instead moves the definition of the
memory allocation interface to a more centralized home in CCCL. RMM will remain as a collection of implementations of
the ``cuda::mr`` interfaces.
