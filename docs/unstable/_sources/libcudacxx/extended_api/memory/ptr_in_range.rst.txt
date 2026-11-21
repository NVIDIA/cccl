.. _libcudacxx-extended-api-memory-ptr_in_range:

``cuda::ptr_in_range``
======================

Defined in the header ``<cuda/memory>``.

.. code:: cuda

   namespace cuda {

   template <typename T>
   [[nodiscard]] __host__ __device__ constexpr
   bool ptr_in_range(T* ptr, T* start, T* end) noexcept;

   } // namespace cuda

Checks whether ``ptr`` lies inside the half-open interval ``[start, end)``.

**Template parameters**

- ``T``: The type of the pointer.

**Parameters**

- ``ptr``: The pointer being tested.
- ``start``: Pointer to the first element in the range.
- ``end``: Pointer to one past the last element in the range.

**Return value**

- ``true`` when the pointer lies in ``[start, end)``, ``false`` otherwise.

**Preconditions**

- ``end`` must be greater than or equal to ``start``.

Example
-------

.. code:: cuda

    #include <cuda/memory>

    __global__ void kernel(float* data, size_t count) {
        float* first = data;
        float* last  = data + count;

        float* elem_ptr = data + threadIdx.x;
        if (cuda::ptr_in_range(elem_ptr, first, last)) {
            *elem_ptr = static_cast<float>(threadIdx.x);
        }
    }

    int main() {
        size_t N          = 32;
        float* device_ptr = nullptr;
        cudaMalloc(&device_ptr, N * sizeof(float));

        kernel<<<1, N>>>(device_ptr, N);
        cudaDeviceSynchronize();

        cudaFree(device_ptr);
        return 0;
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/sMz76hGEc>`__
