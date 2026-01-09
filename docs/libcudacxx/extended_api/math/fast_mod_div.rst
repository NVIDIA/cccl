.. _libcudacxx-extended-api-math-fast-mod-div:

``cuda::fast_mod_div``
======================

Defined in the ``<cuda/cmath>`` header.

.. code:: cuda

    namespace cuda {

    template <typename T, bool DivisorIsNeverOne = false>
    class fast_mod_div {
    public:
        fast_mod_div() = delete;

        __host__ __device__
        explicit fast_mod_div(T divisor) noexcept;

        template <typename U>
        [[nodiscard]] __host__ __device__ friend
        cuda:::std::common_type_t<T, U> operator/(U dividend, fast_mod_div<T> divisor) noexcept;

        template <typename U>
        [[nodiscard]] __host__ __device__ friend
        cuda:::std::common_type_t<T, U> operator%(U dividend, fast_mod_div<T> divisor) noexcept;

        [[nodiscard]] __host__ __device__
        operator T() const noexcept;
    };

    } // namespace cuda

.. code:: cuda

    namespace cuda {

    template <typename T, typename U>
    [[nodiscard]] __host__ __device__
    cuda::std::pair<T, U> div(T dividend, fast_mod_div<U> divisor) noexcept;

    } // namespace cuda

The class ``fast_mod_div`` is used to pre-compute the modulo and division of an integer value, to be used in a second stage for efficiency: :math:`floor\left(\frac{dividend}{divisor}\right)`.

**Parameters**

- ``divisor``:  The divisor.
- ``dividend``: The dividend.
- ``DivisorIsNeverOne``: Indicates that ``divisor != 1`` and skips one comparison in the second stage.

**Constraints**

- ``T`` and ``U`` are integer types of size up to 64-bits.
- ``max_value(dividend type) <= max_value(divisor type)``.

**Preconditions**

- ``divisor > 0``.
- ``dividend >= 0``.
- ``divisor > 1`` if ``DivisorIsNeverOne == true``.

**Performance considerations**

- ``fast_mod_div`` needs to be initialized on the host and executed on the device for optimal performance.
- ``T`` signed type ensures the best performance. ``T == int`` translates to ``SEL``, ``IMAD``, and x2 ``SHF`` instructions.
- 64-bit types are in general slower than 32-bit types.
- ``DivisorIsNeverOne == true`` can be used to skip one comparison.
- ``__builtin_assume(dividend != cuda::std::numeric_limits<U>::max())`` can be used to skip one comparison with unsigned values.

Example
-------

.. code:: cuda

    #include <cuda/cmath>
    #include <cuda/std/cassert>

    __global__ void div_kernel(cuda::fast_mod_div<int> divisor) {
        assert(45 / divisor == 2);
        assert(45 % divisor == 5);
        assert((cuda::div(45, divisor) == cuda::std::pair{2, 5}));
    }

    int main() {
        cuda::fast_mod_div<int> divisor(20);
        div_kernel<<<1, 1>>>(divisor);
        cudaDeviceSynchronize();
        return 0;
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/fM7E9v9aP>`__
