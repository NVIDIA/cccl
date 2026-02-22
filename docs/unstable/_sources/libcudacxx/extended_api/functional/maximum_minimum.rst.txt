.. _libcudacxx-extended-api-functional-maximum-minimum:

``cuda::maximum`` and ``cuda::minimum``
=======================================

.. code:: cuda

    template <typename T>
    struct maximum {
        [[nodiscard]] __host__ __device__ inline
        T operator()(T a, T b) const;
    };

    template <>
    struct maximum<void> {
        template <typename T1, typename T2>
        [[nodiscard]] __host__ __device__ inline
        cuda::std::common_type_t<T1, T2> operator()(T1 a, T2 b) const;
    };

    template <typename T>
    struct minimum {
        [[nodiscard]] __host__ __device__ inline
        T operator()(T a, T b) const;
    };

    template <>
    struct minimum<void> {
        template <typename T1, typename T2>
        [[nodiscard]] __host__ __device__ inline
        cuda::std::common_type_t<T1, T2> operator()(T1 a, T2 b) const;
    };

Function objects for performing maximum and minimum. The functions behave as ``noexcept`` when the comparison between the values is also ``noexcept``.

.. note::

    Differently from ``std::plus`` and other functional operators, ``cuda::maximum`` and ``cuda::minimum`` specialized for ``void`` returns ``cuda::std::common_type_t`` and not the implicit promotion

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
