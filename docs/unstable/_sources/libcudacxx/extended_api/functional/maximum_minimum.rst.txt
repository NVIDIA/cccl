.. _libcudacxx-extended-api-functional-maximum-minimum:

``cuda::maximum`` and ``cuda::minimum``
=======================================

Defined in the header ``<cuda/functional>``.

.. code:: cuda

    template <typename T>
    struct maximum {
        [[nodiscard]] __host__ __device__ constexpr
        T operator()(const T& a, const T& b) const noexcept(/* see below */);
    };

    template <>
    struct maximum<void> {
        template <typename T1, typename T2>
        [[nodiscard]] __host__ __device__ constexpr
        cuda::std::common_type_t<T1, T2> operator()(const T1& a, const T2& b) const noexcept(/* see below */);
    };

    template <typename T>
    struct minimum {
        [[nodiscard]] __host__ __device__ constexpr
        T operator()(const T& a, const T& b) const noexcept(/* see below */);
    };

    template <>
    struct minimum<void> {
        template <typename T1, typename T2>
        [[nodiscard]] __host__ __device__ constexpr
        cuda::std::common_type_t<T1, T2> operator()(const T1& a, const T2& b) const noexcept(/* see below */);
    };

Function objects for performing maximum and minimum operations. The ``operator()`` is ``noexcept`` when the comparison between the values is also ``noexcept``.

.. note::

    Differently from ``std::plus`` and other functional operators, ``cuda::maximum`` and ``cuda::minimum`` specialized for ``void`` returns ``cuda::std::common_type_t`` and not the implicit promotion

Floating-Point Behavior
-----------------------

For floating-point types (and extended floating-point types), ``cuda::maximum`` uses ``cuda::std::fmax`` and ``cuda::minimum`` uses ``cuda::std::fmin`` instead of the comparison operator, following the ``std::fmax``/``std::fmin`` specification for handling special values such as ``NaN``.

This also makes ``cuda::maximum`` and ``cuda::minimum`` commutative for floating-point types, unlike a plain comparison-based approach.

Example
-------

.. code:: cuda

    #include <cuda/functional>
    #include <cuda/std/cstdint>
    #include <cstdio>
    #include <numeric>

    __global__ void maximum_minimum_kernel() {
        uint16_t v1 = 7;
        uint16_t v2 = 3;
        printf("%d\n", cuda::maximum<uint16_t>{}(v1, v2)); // print "7" (uint16_t)
        printf("%d\n", cuda::minimum{}(v1, v2));           // print "3" (int)
    }

    int main() {
        maximum_minimum_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
        int array[] = {3, 7, 5, 2};
        printf("%d\n", std::accumulate(array, array + 4, 0, cuda::maximum{})); // 7
        return 0;
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/44fdTerre>`_
