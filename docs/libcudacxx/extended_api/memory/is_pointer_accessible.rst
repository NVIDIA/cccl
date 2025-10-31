.. _libcudacxx-extended-api-memory-is_pointer_accessible:

``cuda::is_host_accessible``, ``cuda::is_device_accessible``, ``cuda::is_managed``
==================================================================================

Defined in ``<cuda/memory>`` header.

.. code:: cuda

   namespace cuda {

   template <typename Pointer>
   [[nodiscard]] __host__
   bool is_host_accessible(Pointer ptr);

   } // namespace cuda

Determines whether the memory referenced by ``ptr`` is accessible from the host.

----

.. code:: cuda

   namespace cuda {

   template <typename Pointer>
   [[nodiscard]] __host__
   bool is_device_accessible(Pointer ptr, device_ref device);

   } // namespace cuda

Determines whether the memory referenced by ``ptr`` is accessible from the specified ``device``. The function also checks if the memory is peer accessible or backed by a memory pool that is accessible from the specified ``device``.

----

.. code:: cuda

   template <typename Pointer>
   [[nodiscard]] __host__
   bool is_managed(Pointer ptr);

Determines whether the memory referenced by ``ptr`` is backed by Unified Memory (managed memory).

----

**Parameters**

- ``ptr``: A contiguous iterator or pointer that denotes the memory location to query.
- ``device``: A ``device_ref`` object that denotes the device to query.

**Return value**

- ``true`` if the queried property (host access, device access, or managed allocation) can be verified or the memory space cannot be proven, ``false`` otherwise.

.. note::

  The following cases are handled conservatively because the memory space cannot be proven:

  - ``NULL`` pointer.
  - Stack-allocated host memory.
  - ``new`` or ``malloc()``-allocated host memory (not pinned).
  - Global host array or variable.
  - Global ``__device__`` array or variable without retrieving the address with ``cudaGetSymbolAddress()``.

**Constraints**

- ``Pointer`` must be a contiguous iterator or pointer.

**Prerequisites**

- The functions are available only when the CUDA Toolkit is available.

**Exceptions**

- The functions throw a ``cuda::cuda_error`` if the underlying driver API call fails. Note that this function may also return error codes from previous, asynchronous launches.

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
