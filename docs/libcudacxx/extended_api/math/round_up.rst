.. _libcudacxx-extended-api-math-round-up:

``round_up`` Round to the next multiple
=======================================

.. code:: cuda

   template <typename T, typename = U>
   [[nodiscard]] __host__ __device__ inline
   constexpr cuda::std::common_type_t<T, U> round_up(T value, U base_multiple) noexcept;

``value``: The value to be rounded up.
``base_multiple``:  The base multiple to which the value rounds up.

- *Requires*: ``T`` and ``U`` are integral types (including 128-bit integers) or enumerators.
- *Preconditions*: ``a >= 0`` is true and ``b > 0`` is true.
- *Returns*: ``a`` rounded up to the smallest multiple of ``b`` greater than or equal to ``a``. If ``a`` is already a multiple of ``b``, return ``a``.
- *Note*: the result can overflow if ``ceil(a / b) * b`` exceeds the maximum value of the common type of
          ``a`` and ``b``. The condition is checked in debug mode.

.. note::

   The function requires C++17 onwards

**Performance considerations**:

- The function performs a ceiling division (``cuda::ceil_div()``) followed by a multiplication

**Example**:

.. code:: cuda

   #include <cuda/cmath>

   __global__ void example_kernel(int a, unsigned b, unsigned* result) {
     // a = 7, b = 3 -> result = 9
     *result = cuda::round_up(a, b);
   }

`See it on Godbolt TODO`
