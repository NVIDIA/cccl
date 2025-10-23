.. _libcudacxx-extended-api-memory-is_pointer_accessible:

``cuda::is_host_accessible``, ``cuda::is_device_accessible``, ``cuda::is_managed_pointer``
==========================================================================================

Defined in ``<cuda/memory>`` header.

.. code:: cuda

   namespace cuda {

   template <typename Pointer>
   [[nodiscard]] __host__
   bool is_host_accessible(Pointer ptr) noexcept;

   } // namespace cuda

Determines whether the memory referenced by ``ptr`` is accessible from the host.

----

.. code:: cuda

   namespace cuda {

   template <typename Pointer>
   [[nodiscard]] __host__
   bool is_device_accessible(Pointer ptr, device_ref device) noexcept;

   } // namespace cuda

Determines whether the memory referenced by ``ptr`` is accessible from the active CUDA device.

----

.. code:: cuda

   template <typename Pointer>
   [[nodiscard]] __host__
   bool is_managed_pointer(Pointer ptr) noexcept;

Determines whether the memory referenced by ``ptr`` is backed by Unified Memory (managed memory).

----

**Parameters**

- ``ptr``: A contiguous iterator or pointer that denotes the memory location to query.

**Return value**

- ``true`` if the queried property (host access, device access, or managed allocation) can be confirmed, ``false`` otherwise.

**Constraints**

- ``Pointer`` must be a contiguous iterator or pointer.

**Prerequisites**

- When the call is evaluated in a constant-evaluated context, the functions conservatively return ``true`` because driver queries cannot be performed.
- The functions are available only when the CUDA Toolkit is available.

.. note::

  The following cases are handled conservatively because the memory space cannot be proven:

  - ``NULL`` pointer.
  - Stack-allocated host memory.
  - ``new`` or ``malloc()``-allocated host memory (not pinned).
  - Global host array or variable.
  - Global ``__device__`` array or variable without retrieving the address with ``cudaGetSymbolAddress()``.

Example
-------

.. code:: cuda

    #include <cassert>
    #include <cuda/memory>

    int main() {
        void* host_ptr    = nullptr;
        void* device_ptr  = nullptr;
        void* managed_ptr = nullptr;

        cudaMallocHost(&host_ptr, 1024);
        cudaMalloc(&device_ptr, 1024);
        cudaMallocManaged(&managed_ptr, 1024);

        assert(cuda::is_host_accessible(host_ptr));
        assert(!cuda::is_device_accessible(host_ptr));

        assert(cuda::is_device_accessible(device_ptr));
        assert(!cuda::is_host_accessible(device_ptr));

        assert(cuda::is_host_accessible(managed_ptr));
        assert(cuda::is_device_accessible(managed_ptr));
        assert(cuda::is_managed_pointer(managed_ptr));

        cudaFreeHost(host_ptr);
        cudaFree(device_ptr);
        cudaFree(managed_ptr);
    }
