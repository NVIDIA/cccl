.. _libcudacxx-extended-api-memory-pointer_in_range:

``cuda::pointer_in_range``
=========================

.. code:: cuda

   template <typename T>
   [[nodiscard]] __host__ __device__ inline
   T* pointer_in_range(T* ptr, T* start, T* end) noexcept

Checks whether ``ptr`` lies inside the half-open interval ``[start, end)``. The interval bounds are interpreted using the standard pointer comparison semantics for ``T*``.

**Template parameters**

- ``T``: The type of the pointer being tested.

**Parameters**

- ``ptr``: The pointer being tested.
- ``start``: Pointer to the first element in the range.
- ``end``: Pointer one past the last element in the range.

**Return value**

- ``true`` when the pointer lies in ``[start, end)``, ``false`` otherwise.

**Preconditions**

- ``end`` must be strictly greater than ``start``.

Example
-------

.. code:: cuda

    #include <cuda/memory>

    __global__ void kernel(float* data, size_t count) {
        float* first = data;
        float* last  = data + count;

        float* elem_ptr = data + threadIdx.x;
        if (cuda::pointer_in_range(elem_ptr, first, last)) {
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

`See it on Godbolt 🔗 <https://godbolt.org/z/EPfMErjGK>`_
