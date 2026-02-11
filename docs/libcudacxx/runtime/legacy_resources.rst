.. _cccl-runtime-legacy-resources:
.. _libcudacxx-extended-api-memory-resources-legacy-resources:

Legacy resources
================

Legacy memory resources provide synchronous allocation interfaces backed by the CUDA Runtime's legacy allocation APIs.
They are primarily intended for compatibility with older toolkits or platforms that do not support the newer memory
pool-based resources. Prefer the modern memory resources where available.

For the full memory resource model and property system, see
:ref:`Memory Resources (Extended API) <libcudacxx-extended-api-memory-resources>`.

``cuda::mr::legacy_pinned_memory_resource``
-------------------------------------------
.. _libcudacxx-memory-resource-legacy-pinned-memory-resource:

Provides pinned (page-locked) host allocations using ``cudaMallocHost`` and ``cudaFreeHost``. This resource is
*synchronous-only* and is intended as a compatibility fallback. For CUDA 12.6 and later, prefer
``cuda::pinned_memory_resource``.

.. code:: cpp

   #include <cuda/memory_resource>

   void use_legacy_pinned() {
     cuda::mr::legacy_pinned_memory_resource resource{};
     void* ptr = resource.allocate_sync(1024, 64);
     // Use memory...
     resource.deallocate_sync(ptr, 1024, 64);
   }

``cuda::mr::legacy_managed_memory_resource``
--------------------------------------------
.. _libcudacxx-memory-resource-legacy-managed-memory-resource:

Provides managed (unified) allocations using ``cudaMallocManaged`` and ``cudaFree``. This resource is
*synchronous-only* and accepts the CUDA attachment flags (``cudaMemAttachGlobal`` / ``cudaMemAttachHost``). Prefer
``cuda::managed_memory_resource`` when available.

.. code:: cpp

   #include <cuda/memory_resource>

   void use_legacy_managed() {
     cuda::mr::legacy_managed_memory_resource resource{cudaMemAttachGlobal};
     void* ptr = resource.allocate_sync(1024, 64);
     // Use memory...
     resource.deallocate_sync(ptr, 1024, 64);
   }
