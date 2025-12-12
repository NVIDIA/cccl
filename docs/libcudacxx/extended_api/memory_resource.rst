.. _libcudacxx-extended-api-memory-resources:

Memory Resources
================

.. toctree::
   :hidden:
   :maxdepth: 1

   memory_resource/properties
   Resources <memory_resource/resource>
   Resource wrapper <memory_resource/resource_ref>

The ``<cuda/memory_resource>`` header provides a standard C++ interface for *heterogeneous*, *stream-ordered* memory
allocation tailored to the needs of CUDA C++ developers. This design builds off of the success of the `RAPIDS Memory Manager (RMM) <https://github.com/rapidsai/rmm>`__
project and evolves the design based on lessons learned.

``<cuda/memory_resource>`` is not intended to replace RMM, but instead moves the definition of the memory allocation
interface to a more centralized home in CCCL. RMM will remain as a collection of implementations of the ``cuda::mr``
interfaces.

At a high level, the header provides:

.. list-table::
   :widths: 30 50 20
   :header-rows: 0

   * - :ref:`cuda::get_property <libcudacxx-extended-api-memory-resources-properties>`
     - Infrastructure to tag a user defined type with a given property
     - stable CCCL 3.1.0 / CUDA 13.1, experimental CCCL 2.2.0 / CUDA 12.3
   * - :ref:`cuda::mr::{synchronous_}resource <libcudacxx-extended-api-memory-resources-resource>` and
       :ref:`cuda::mr::{synchronous_}resource_with <libcudacxx-extended-api-memory-resources-resource>`
     - Concepts that provide proper constraints for arbitrary memory resources.
     - stable CCCL 3.1.0 / CUDA 13.1, experimental CCCL 2.2.0 / CUDA 12.3
   * - :ref:`cuda::mr::{synchronous_}resource_ref <libcudacxx-extended-api-memory-resources-resource-ref>`
     - A non-owning type-erased memory resource wrapper that enables consumers to specify properties of resources that they expect.
     - stable CCCL 3.2.0 / CUDA 13.2, experimental CCCL 2.2.0 / CUDA 12.3

These features are an evolution of `std::pmr::memory_resource <https://en.cppreference.com/w/cpp/header/memory_resource>`__
that was introduced in C++17. While ``std::pmr::memory_resource`` provides a polymorphic memory resource that can be
adopted through inheritance, it is not properly suited for heterogeneous systems.

With the current design it ranges from cumbersome to impossible to verify whether a memory resource provides allocations
that are e.g. accessible on device, or whether it can utilize other allocation mechanisms.

To better support asynchronous CUDA `stream-ordered allocations <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-ordered-memory-allocator>`__
libcu++ provides :ref:`cuda::stream_ref <libcudacxx-extended-api-streams-stream-ref>` as a wrapper around
``cudaStream_t``. The definition of ``cuda::stream_ref`` can be found in the ``<cuda/stream>`` header.
