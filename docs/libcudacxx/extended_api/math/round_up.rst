.. _libcudacxx-extended-api-math-round-up:

``round_up`` Round to the next multiple
=======================================

.. code:: cuda

   template <typename T, typename = U>
   [[nodiscard]] __host__ __device__ constexpr decltype(T{} / U{}) round_up(T a, U b) noexcept;

- *Requires*: ``T`` and ``U`` are integral types or enumerators.
- *Preconditions*: ``a >= 0`` is true and ``b > 0`` is true.
- *Returns*: ``a`` rounded to the next multiple of ``b``. If ``a`` is already a multiple of ``b``, return ``a``.
- *Note*: the result can overflow if ``ceil(a / b) * b`` exceeds the maximum value of the common type of
          ``a`` and ``b``. The condition is checked in debug mode.

.. note::

   The function requires C++17 onwards

**Example**:

.. code:: cuda

   #include <cuda/cmath>

   __global__ void example_kernel(int a, unsigned b, unsigned* result) {
     // a = 7, b = 3 -> result = 9
     *result = cuda::round_up(a, b);
   }

`See it on Godbolt TODO`
