.. _libcudacxx-extended-api-memory-get-device-address:

``cuda::get_device_address``
============================

Defined in the headers ``<cuda/memory>`` and ``<cuda/functional>``.

.. code:: cuda

  namespace cuda {

  template <typename T>
  [[nodiscard]] __host__ __device__ inline
  T* get_device_address(T& device_object);                    // (1)

  template <typename T>
  [[nodiscard]] __host__ inline
  T* get_device_address(T& device_object, device_ref device); // (2)

  } // namespace cuda

``cuda::get_device_address`` returns a valid pointer to a device object for the current (1) or ``device`` (2) device. It replaces uses of ``cudaGetSymbolAddress``, which requires an inout parameter.

**Parameters**

- ``device_object``: Reference to a device object. (1, 2)
- ``device``: Device for which the object's address shall be retrieved. (2)

**Constraints**

- ``device_object`` must be a ``__device__`` or ``__constant__`` decorated variable.

Example
-------

.. code:: cuda

  #include <cuda/devices>
  #include <cuda/memory>

  __device__ int device_object[] = {42, 1337, -1, 0};

  __global__ void example_kernel(int *data) { ... }

  void example()
  {
    cuda::device_ref device{0};

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
      T* device_address = cuda::get_device_address(device_object, device);

      cudaPointerAttributes attributes;
      cudaError_t status = cudaPointerGetAttributes(&attributes, device_address);
      assert(status == cudaSuccess);
      assert(attributes.devicePointer == device_address);

      // Safe to call a kernel
      example_kernel<<<1, 1>>>(device_address);
    }
  }
