.. _libcudacxx-extended-api-math-ceil-div:

``cuda::ceil_div``
==================

.. code:: cuda

   template <typename T, typename U>
   [[nodiscard]] __host__ __device__ inline constexpr
   cuda::std::common_type_t<T, U> ceil_div(T value, U divisor) noexcept;

The function computes the ceiling division between two integral or enumerator values :math:`ceil(\frac{value}{base\_multiple})`.

**Parameters**

- ``value``: The value to be divided.
- ``divisor``: The divisor.

**Return value**

Divides ``value`` by ``divisor``. If ``value`` is not a multiple of ``divisor`` rounds the result up to the next integer value.

**Constraints**

- ``T`` and ``U`` are integer types or enumerators.

**Preconditions**

- ``value >= 0``
- ``divisor > 0``

**Performance considerations**

- The function computes ``(value + divisor - 1) / divisor`` when the common type is a signed integer.
- The function computes ``min(value, 1 + ((value - 1) / divisor)`` when the common type is an unsigned integer in CUDA, which generates less instructions than ``(value / divisor) + ((value / divisor) * divisor != value)``, especially for 64-bit types.

Example
-------

This API is very useful for determining the *number of thread blocks* required to process a fixed amount of work, given a fixed number of threads per block:

.. code:: cuda

    #include <cuda/cmath>
    #include <cuda/std/span>
    #include <thrust/device_vector.h>

    __global__ void vector_scale_kernel(cuda::std::span<float> span, float scale) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < span.size())
            span[index] *= scale;
    }

    int main() {
        int   num_items = 100'000;
        float scale = 2.f;
        thrust::device_vector<float> d_vector(num_items, 1.f);
        // Given a fixed number of threads per block...
        constexpr int threads_per_block = 256;
        // ...dividing some "n" by "threads_per_block" may lead to a remainder,
        // requiring the kernel to be launched with an extra thread block to handle it.
        auto num_thread_blocks = cuda::ceil_div(num_items, threads_per_block);
        auto d_ptr             = thrust::raw_pointer_cast(d_vector.data());
        cuda::std::span<float> d_span(d_ptr, num_items);

        vector_scale_kernel<<<num_thread_blocks, threads_per_block>>>(d_span, scale);
        cudaDeviceSynchronize();
        return 0;
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/hbxscWGT9>`_
