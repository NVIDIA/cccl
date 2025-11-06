.. _libcudacxx-extended-api-memory-is_pointer_accessible:

``cuda::is_host_accessible``, ``cuda::is_device_accessible``, ``cuda::is_managed``
==================================================================================

Defined in ``<cuda/memory>`` header.

.. code:: cuda

   namespace cuda {

   [[nodiscard]] inline
   bool is_host_accessible(const void* ptr);

   } // namespace cuda

Determines whether the memory referenced by ``ptr`` is accessible from the host.

----

.. code:: cuda

   namespace cuda {

   [[nodiscard]] inline
   bool is_device_accessible(const void* ptr, device_ref device);

   } // namespace cuda

Determines whether the memory referenced by ``ptr`` is accessible from the specified ``device``. The function also checks if the memory is peer accessible or backed by a memory pool that is accessible from the specified ``device``.

----

.. code:: cuda

   namespace cuda {

   [[nodiscard]] inline
   bool is_managed(const void* ptr);

   } // namespace cuda

Determines whether the memory referenced by ``ptr`` is backed by Unified Memory (managed memory).

----

**Parameters**

- ``ptr``: A pointer that denotes the memory location to query.
- ``device``: A ``device_ref`` object that denotes the device to query.

**Return value**

- ``true`` if the queried property (host access, device access, or managed allocation) can be verified, ``false`` otherwise.

.. note::

  The functions cannot correctly determine the accessibility of a ``__device__`` global array pointer with an offset, for example ``ptr + 1``.

**Prerequisites**

- The functions are available only when the CUDA Toolkit is available.

**Exceptions**

- The functions throw a ``cuda::cuda_error`` if the underlying driver API calls fail. Note that this function may also fail with error codes from previous, asynchronous launches.

**Undefined Behavior**

- The functions have undefined behavior if the pointer is not valid, for example an already freed pointer.

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
