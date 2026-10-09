.. _libcudacxx-extended-api-math-abs-diff:

``cuda::abs_diff``
========================

Defined in the ``<cuda/cmath>`` header.

.. code:: cuda

   namespace cuda {

   template <class T>
   [[nodiscard]] __host__ __device__ constexpr
   cuda::std::make_unsigned_t<T> abs_diff(T lhs, T rhs) noexcept;

   } // namespace cuda

The function ``cuda::abs_diff`` computes absolute difference of two integers and returns the result as an *unsigned* value.

**Parameters**

- ``lhs``: The left-hand side input.
- ``rhs``: The right-hand side input.

**Return value**

Returns a ``cuda::std::make_unsigned_t<T>`` value of ``lhs`` and ``rhs`` absolute difference.

**Constraints**

- ``T`` must be an integer type.

**Performance considerations**

- On device, for types whose width is smaller than 64-bit results in:

  - ``VABSDIFF`` on ``SM75``, ``SM80``, ``SM86``, ``SM87``, ``SM89``, ``SM90``, ``SM100``, ``SM103``, and ``SM110``.

- Other target/type combinations result in ``max(lhs, rhs) - min(lhs, rhs)`` equivalent.

Example
-------

.. code:: cuda

    #include <cuda/cmath>
    #include <cuda/std/cassert>
    #include <cuda/std/cstdint>
    #include <cuda/std/limits>

    __global__ void kernel()
    {
        auto lhs = cuda::std::numeric_limits<cuda::std::int32_t>::min();
        auto rhs = cuda::std::numeric_limits<cuda::std::int32_t>::max();

        assert(cuda::abs_diff(lhs, rhs) == cuda::std::numeric_limits<cuda::std::uint32_t>::max());
    }

    int main()
    {
        kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/xfv6EaG5e>`__
