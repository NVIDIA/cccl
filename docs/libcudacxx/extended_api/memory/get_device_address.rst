.. _libcudacxx-extended-api-memory-get-device-address:

``cuda::get_device_address``
============================

Defined in the headers ``<cuda/memory>`` and ``<cuda/functional>``.

.. code:: cuda

   namespace cuda {

   [[nodiscard]] __host__ __device__
   T* get_device_address(T& __device_object) noexcept;

   } // namespace cuda

``cuda::get_device_address`` returns a valid pointer to a device object.
It replaces uses of ``cudaGetSymbolAddress``, which requires an inout parameter.

Example
-------

.. code:: cuda

  #include <cuda/memory>

  __device__ int device_object[] = {42, 1337, -1, 0};

  __global__ void example_kernel(int *data) { ... }

  void example()
  {
    {
      T* host_address = cuda::std::addressof(device_object);

      cudaPointerAttributes attributes;
      cudaError_t status = cudaPointerGetAttributes(&attributes, host_address);
      assert(status == cudaSuccess);
      assert(attributes.devicePointer == nullptr);

      // Calling a kernel with host_address would segfault
      // example_kernel<<<1, 1>>>(host_address);
    }

    {
      T* device_address = cuda::get_device_address(device_object);

      cudaPointerAttributes attributes;
      cudaError_t status = cudaPointerGetAttributes(&attributes, device_address);
      assert(status == cudaSuccess);
      assert(attributes.devicePointer == device_address);

      // Safe to call a kernel
      example_kernel<<<1, 1>>>(device_address);
    }
  }
