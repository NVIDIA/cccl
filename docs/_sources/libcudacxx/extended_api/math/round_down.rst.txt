.. _libcudacxx-extended-api-math-round-down:

``cuda::round_down``
====================

.. code:: cuda

   template <typename T, typename U>
   [[nodiscard]] __host__ __device__ inline constexpr
   cuda::std::common_type_t<T, U> round_down(T value, U base_multiple) noexcept;

The function computes the round down to the largest multiple of an integral or enumerator value :math:`floor(\frac{value}{base\_multiple}) * base\_multiple`

**Parameters**

- ``value``: The value to be rounded down.
- ``base_multiple``:  The base multiple to which the value rounds down.

**Return value**

``value`` rounded down to the largest multiple of ``base_multiple`` less than or equal to ``value``. If ``value`` is already a multiple of ``base_multiple``, returns ``value``.

**Constraints**

- ``T`` and ``U`` are integer types or enumerators.

**Preconditions**

- ``value >= 0``
- ``base_multiple > 0``

**Performance considerations**

- The function performs a truncation division followed by a multiplication. It provides better performance than ``(value / base_multiple) * base_multiple`` when the common type is a signed integer

Example
-------

.. code:: cuda

    #include <cuda/cmath>
    #include <cstdio>

    __global__ void round_up_kernel() {
        int      value    = 7;
        unsigned multiple = 3;
        printf("%d\n", cuda::round_down(value, multiple)); // print "6"
    }

    int main() {
        round_up_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
        return 0;
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/9vcxo3d8j>`_
