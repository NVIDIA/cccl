.. _libcudacxx-extended-api-math-min-max:

``cuda::min`` and ``cuda::max``
===============================

Defined in the ``<cuda/cmath>`` header.

.. code:: cuda

   namespace cuda {

   template <typename T>
   [[nodiscard]] __host__ __device__ constexpr
   T min(T a, T b) noexcept;

   template <typename T>
   [[nodiscard]] __host__ __device__ constexpr
   T max(T a, T b) noexcept;

   } // namespace cuda

The functions return the smaller and the larger of two values, respectively. Contrary to ``cuda::std::min`` and ``cuda::std::max``, which take their arguments by reference and return a reference to one of them, the functions take and return values. This lets the compiler compute the result in registers, instead of selecting the address of an argument and loading the result from it.

For floating-point types, the values are compared like with ``cuda::std::fmin`` and ``cuda::std::fmax``.

**Parameters**

- ``a``, ``b``: The values to compare.

**Return value**

- ``min``: The smaller of ``a`` and ``b``.
- ``max``: The larger of ``a`` and ``b``.
- For floating-point types, if one of the values is NaN, the result is the other value. If both values are NaN, the result is NaN.

**Constraints**

- ``T`` is an arithmetic type or an extended floating-point type.

**Performance considerations**

- On device, the functions map to a single ``min`` or ``max`` instruction for integer types of up to 64 bits and ``float``, for ``double`` on GPUs with fast FP64 arithmetic, and for ``__half`` and ``__nv_bfloat16`` on ``SM80`` and newer. On GPUs with slow FP64 arithmetic, ``double`` values are compared with integer instructions instead.

Example
-------

.. code:: cuda

    #include <cuda/cmath>
    #include <cuda/std/cassert>
    #include <cuda/std/limits>

    __global__ void kernel()
    {
        assert(cuda::min(1, 2) == 1);
        assert(cuda::max(1, 2) == 2);

        const float nan = cuda::std::numeric_limits<float>::quiet_NaN();
        assert(cuda::min(nan, 1.0f) == 1.0f);
        assert(cuda::max(1.0f, nan) == 1.0f);
    }

    int main()
    {
        kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/v9aK1xx4z>`__
