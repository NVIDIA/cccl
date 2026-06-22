.. _libcudacxx-extended-api-numeric-isclose:

``cuda::isclose``
=================

.. code:: cpp

   template <class T>
   [[nodiscard]] constexpr bool isclose(T lhs, T rhs) noexcept;

   template <class T>
   [[nodiscard]] constexpr bool isclose(T lhs, T rhs, float rel_tol) noexcept;

   template <class T, class AbsTol>
   [[nodiscard]] constexpr bool isclose(T lhs, T rhs, float rel_tol, AbsTol abs_tol) noexcept;

   template <class Complex>
   [[nodiscard]] bool isclose(const Complex& lhs, const Complex& rhs) noexcept;

   template <class Complex>
   [[nodiscard]] bool isclose(const Complex& lhs, const Complex& rhs, float rel_tol) noexcept;

   template <class Complex, class AbsTol>
   [[nodiscard]] bool isclose(const Complex& lhs, const Complex& rhs, float rel_tol, AbsTol abs_tol) noexcept;

``cuda::isclose`` checks whether two values are approximately equal using the weak symmetric comparison described by
`PEP 485 <https://peps.python.org/pep-0485/>`_:

.. code:: cpp

   abs(lhs - rhs) <= max(abs_tol, rel_tol * max(abs(lhs), abs(rhs)))

The overloads without ``abs_tol`` use ``abs_tol == 0``. The overloads without ``rel_tol`` use a default relative
tolerance based on the promoted comparison type:

.. code:: cpp

   pow(10, -ceil_div(cuda::std::numeric_limits<Comparison>::max_digits10, 2))

For ``double`` comparisons, this is ``1e-9``, matching the default relative tolerance from PEP 485. Lower- and
higher-precision comparison types use a correspondingly smaller or larger default.

**Parameters**

- ``lhs``: The first value to compare.
- ``rhs``: The second value to compare.
- ``rel_tol``: The relative tolerance. Must be finite and non-negative. Passing ``0`` performs a purely absolute
  tolerance check when ``abs_tol`` is non-zero.
- ``abs_tol``: The absolute tolerance. Must be finite and non-negative. This is useful for comparisons near zero. The
  supplied type may promote to the value comparison type, but may not make the value comparison type wider.

**Return value**

- Returns ``true`` if ``lhs`` and ``rhs`` are close to each other, otherwise returns ``false``.

**Constraints**

- Scalar overloads require ``lhs`` and ``rhs`` to have the same arithmetic type. ``abs_tol`` must be representable in the
  value comparison type after promotion. For example, ``double`` values may be compared with a ``float`` absolute
  tolerance, but ``float`` values cannot use a ``double`` absolute tolerance. ``rel_tol`` is always a ``float``.
- Complex overloads accept ``cuda::std::complex<T>`` and ``cuda::complex<T>`` operands. ``lhs`` and ``rhs`` must have the
  same complex type, and ``abs_tol`` must be representable in the complex value comparison type after promotion.
  ``rel_tol`` is always a ``float``.

**Special values**

- NaN is never close to any value, including another NaN.
- Infinity and negative infinity are only close to themselves.
- With the default ``abs_tol == 0``, comparisons near zero generally require an explicitly supplied absolute tolerance.

For complex values, ``cuda::isclose`` follows the ``cmath.isclose`` model from PEP 485: the difference and scaling values
are computed from complex magnitudes, rather than comparing the real and imaginary components independently.

Example
-------

.. code:: cuda

    #include <cuda/__complex_>
    #include <cuda/numeric>
    #include <cuda/std/cassert>
    #include <cuda/std/complex>

    __global__ void kernel()
    {
        assert(cuda::isclose(1.0, 1.0 + 5e-10));
        assert(!cuda::isclose(1.0, 1.0 + 5e-8));

        assert(!cuda::isclose(0.0, 1e-12));
        assert(cuda::isclose(0.0, 1e-12, 0.0, 1e-12));

        cuda::std::complex<double> z1{1.0, 1.0};
        cuda::std::complex<double> z2{2.0, 0.0};
        assert(cuda::isclose(z1, z2, 0.75));

        cuda::complex<float> z3{1.0f, 1.0f};
        cuda::complex<float> z4{2.0f, 0.0f};
        assert(cuda::isclose(z3, z4, 0.75f));
    }

    int main()
    {
        kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
    }
