.. _libcudacxx-extended-api-memory-is_pointer_accessible:

``cuda::is_host_accessible``, ``cuda::is_device_accessible``, ``cuda::is_managed``
==================================================================================

Defined in the ``<cuda/memory>`` header.

.. code:: cuda

   namespace cuda {

   [[nodiscard]] inline
   bool is_host_accessible(const void* ptr); // (1)

   [[nodiscard]] inline
   bool is_device_accessible(const void* ptr, device_ref device); // (2)

   [[nodiscard]] inline
   bool is_managed(const void* ptr); // (3)

   } // namespace cuda

Determines whether the memory referenced by ``ptr`` is accessible from the host (1), from the specified ``device`` (2), or is backed by Unified Memory (managed memory) (3).

- ``is_device_accessible()`` also checks whether the memory is peer-accessible or allocated from a memory pool accessible to the specified ``device``.
- ``is_host_accessible()`` also checks whether the memory is allocated from a memory pool accessible to the host.

----

**Parameters**

- ``ptr``: A pointer to the memory location to query.
- ``device``: A ``device_ref`` that denotes the device to query. (2)

**Return value**

- ``true`` if the queried property (host access, device access, or managed allocation) holds; otherwise, ``false``.

.. note::

  A ``__device__`` global array or variable cannot be used directly from host code without first retrieving its address with ``cudaGetSymbolAddress()``.

**Prerequisites**

- The functions are available only when the CUDA Toolkit is available.

**Exceptions**

- The functions throw a ``cuda::cuda_error`` if the underlying driver API calls fail. Note that these functions may also fail with error codes from previously launched asynchronous operations.

**Undefined Behavior**

- The functions have undefined behavior if the pointer is not valid, for example, an already freed pointer.

Example
-------

.. code:: cuda

    #include <cassert>
    #include <cuda/memory>
    #include <cuda_runtime_api.h>

    int main() {
        cuda::device_ref dev{0};
        void* host_ptr    = nullptr;
        void* device_ptr  = nullptr;
        void* managed_ptr = nullptr;

        cudaMallocHost(&host_ptr, 1024);
        cudaMalloc(&device_ptr, 1024);
        cudaMallocManaged(&managed_ptr, 1024);

        assert(cuda::is_host_accessible(host_ptr));
        assert(!cuda::is_device_accessible(host_ptr, dev));

        assert(cuda::is_device_accessible(device_ptr, dev));
        assert(!cuda::is_host_accessible(device_ptr));

        assert(cuda::is_host_accessible(managed_ptr));
        assert(cuda::is_device_accessible(managed_ptr, dev));
        assert(cuda::is_managed(managed_ptr));

        cudaFreeHost(host_ptr);
        cudaFree(device_ptr);
        cudaFree(managed_ptr);
    }
