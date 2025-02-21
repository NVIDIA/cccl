.. _libcudacxx-extended-api-math-round-up:

``round_up`` Round to the next multiple
=======================================

.. code:: cuda

   template <typename T, typename U>
   [[nodiscard]] __host__ __device__ inline
   constexpr cuda::std::common_type_t<T, U> round_up(T value, U base_multiple) noexcept;

The function computes the round up to the smallest multiple of an integral or enumerator value :math:`ceil(\frac{value}{base\_multiple}) * base\_multiple`

**Parameters**

- ``value``: The value to be rounded up.
- ``base_multiple``:  The base multiple to which the value rounds up.

**Return value**

``value`` rounded up to the smallest multiple of ``base_multiple`` greater than or equal to ``value``. If ``value`` is already a multiple of ``base_multiple``, return ``value``.

.. note::

    The result can overflow if ``ceil(value / base_multiple) * base_multiple`` exceeds the maximum value of the common type of ``value`` and ``base_multiple``. The condition is checked in debug mode.

**Preconditions**

- *Compile-time*: ``T`` and ``U`` are integral types (including 128-bit integers) or enumerators.
- *Run-time*: ``value >= 0`` and ``base_multiple > 0``.

**Performance considerations**

- The function performs a ceiling division (``cuda::ceil_div()``) followed by a multiplication

Example
-------

.. code:: cuda

    #include <cuda/cmath>
    #include <cstdio>

    __global__ void round_up_kernel() {
        int      value    = 7;
        unsigned multiple = 3;
        printf("%d\n", cuda::round_up(value, multiple)); // print "9"
    }

    int main() {
        round_up_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
        return 0;
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/9vcxo3d8j>`_
