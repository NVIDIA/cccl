.. _libcudacxx-extended-api-math-round-down:

``round_down`` Round to the previous multiple
=============================================

.. code:: cuda

   template <typename T, typename = U>
   [[nodiscard]] __host__ __device__ inline
   constexpr cuda::std::common_type_t<T, U> round_down(T value, U base_multiple) noexcept;

``value``: The value to be rounded down.
``base_multiple``:  The base multiple to which the value rounds down.

- *Requires*: ``T`` and ``U`` are integral types (including 128-bit integers) or enumerators.
- *Preconditions*: ``a >= 0`` is true and ``b > 0`` is true.
- *Returns*: ``a`` rounded down to the largest multiple of ``b`` less than or equal to ``a``. If ``a`` is already a multiple of ``b``, return ``a``.

.. note::

   The function requires C++17 onwards

**Performance considerations**:

- The function performs a truncation division followed by a multiplication. It provides better performance than ``a / b * b`` when the common type is a signed integer

**Example**:

.. code:: cuda

   #include <cuda/cmath>

   __global__ void example_kernel(int a, unsigned b, unsigned* result) {
     // a = 7, b = 3 -> result = 6
     *result = cuda::round_down(a, b);
   }

`See it on Godbolt TODO`
