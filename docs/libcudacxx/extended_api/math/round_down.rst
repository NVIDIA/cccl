.. _libcudacxx-extended-api-math-round-down:

``round_down`` Round to the previous multiple
=============================================

.. code:: cuda

   template <typename T, typename = U>
   [[nodiscard]] __host__ __device__ constexpr decltype(T{} / U{}) round_down(T a, U b) noexcept;

- *Requires*: ``T`` and ``U`` are integral types or enumerators.
- *Preconditions*: ``a >= 0`` is true and ``b > 0`` is true.
- *Returns*: ``a`` rounded to the previous multiple of ``b``. If ``a`` is already a multiple of ``b``, return ``a``.

.. note::

   The function requires C++17 onwards

**Example**:

.. code:: cuda

   #include <cuda/cmath>

   __global__ void example_kernel(int a, unsigned b, unsigned* result) {
     // a = 7, b = 3 -> result = 6
     *result = cuda::round_down(a, b);
   }

`See it on Godbolt TODO`
