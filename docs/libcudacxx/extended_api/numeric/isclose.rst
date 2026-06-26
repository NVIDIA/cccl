.. _libcudacxx-extended-api-numeric-isclose:

``cuda::isclose``
=================

Defined in ``<cuda/numeric>`` header.

.. code:: cpp

   namespace cuda {

   template <class T>
   [[nodiscard]] __host__ __device__ __tile__ constexpr
    bool isclose(T lhs, T rhs) noexcept;

   template <class T>
   [[nodiscard]] __host__ __device__ __tile__ constexpr
    bool isclose(T lhs, T rhs, float relative_tol) noexcept;

   template <class T>
   [[nodiscard]] __host__ __device__ __tile__ constexpr
   bool isclose(T lhs, T rhs, float relative_tol, T absolute_tol) noexcept;

   template <class Complex>
   [[nodiscard]] __host__ __device__ constexpr
   bool isclose(const Complex& lhs, const Complex& rhs) noexcept;

   template <class Complex>
   [[nodiscard]] __host__ __device__ constexpr
   bool isclose(const Complex& lhs, const Complex& rhs, float relative_tol) noexcept;

   template <class Complex, class AbsTol>
   [[nodiscard]] __host__ __device__ constexpr
   bool isclose(const Complex& lhs, const Complex& rhs,
                float  relative_tol,
                AbsTol absolute_tol) noexcept;

   } // namespace cuda

``cuda::isclose`` checks whether two values are approximately equal using the weak symmetric comparison in a similar manner to `PEP 485 <https://peps.python.org/pep-0485/>`_:

.. code:: cpp

   abs(lhs - rhs) <= max(absolute_tol, relative_tol * max(abs(lhs), abs(rhs)))

- The overloads without ``absolute_tol`` use ``absolute_tol == 0``.
- The overloads without ``relative_tol`` use a default relative tolerance based on half of available digits of accuracy.

**Parameters**

- ``lhs``: The first value to compare.
- ``rhs``: The second value to compare.
- ``relative_tol``: The relative tolerance. Passing ``0`` performs a purely absolute tolerance check when ``absolute_tol`` is non-zero.
- ``absolute_tol``: The absolute tolerance. This is useful for comparisons near zero.

**Precision**

- ``relative_tol``:Must be in the range [0.0, 1.0].
- ``absolute_tol``: Must be finite and non-negative.

**Return value**

- Returns ``true`` if ``lhs`` and ``rhs`` are close to each other, otherwise returns ``false``.

**Constraints**

- Scalar overloads require ``lhs``, ``rhs``, ``absolute_tol`` to have the same arithmetic type (integer or floating point).
- Complex overloads accept ``cuda::std::complex<T>``, ``cuda::complex<T>``, ``std::complex<T>`` operands.
- ``AbsTol`` must be the same type as the complex value type.

**Special values**

- ``NaN`` is never close to any value, including another ``NaN``.
- Infinity and negative infinity are only close to themselves.

Example
-------

.. code:: cuda

    #include <cuda/numeric>
    #include <cuda/std/cassert>
    #include <cuda/std/complex>

    __global__ void kernel()
    {
        assert(cuda::isclose( 1.0, 1.0 + 5e-10));
        assert(!cuda::isclose(1.0, 1.0 + 5e-8));

        assert(!cuda::isclose(0.0, 1e-12));
        assert(cuda::isclose( 0.0, 1e-12, 0.0, 1e-12));

        cuda::std::complex<double> z1{1.0, 1.0};
        cuda::std::complex<double> z2{2.0, 0.0};
        assert(cuda::isclose(z1, z2, 0.75));
    }

    int main()
    {
        kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
    }
