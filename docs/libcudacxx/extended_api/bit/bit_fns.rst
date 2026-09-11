.. _libcudacxx-extended-api-bit-bit_fns:

``cuda::bit_fns``
=================

Defined in the ``<cuda/bit>`` header.

.. code:: cuda

   namespace cuda {

   template <typename T>
   [[nodiscard]] __host__ __device__ __tile__ constexpr
   int bit_fns(T value, int rank) noexcept;

   } // namespace cuda

The function finds the ``rank``-th set bit of ``value``, counting set bits from the least significant one, and returns its position. Both ``rank`` and the returned position are zero-based. If ``value`` has fewer than ``rank + 1`` set bits the function returns ``-1``.

**Parameters**

- ``value``: The unsigned integer value to search.
- ``rank``:  The zero-based rank of the set bit to find.

.. warning::

    Unlike the `CUDA Math function <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__INT.html#group__cuda__math__intrinsic__int_1ga2fc8e909eb9a959dcc3262e54365bfc5>`__ ``__fns``, which uses one-based ranks (``offset``), ``cuda::bit_fns`` uses zero-based ranks.

**Return value**

- The zero-based position of the set bit with rank ``rank``, or ``-1`` if ``value`` has fewer than ``rank + 1`` set bits.

**Constraints**

- ``T`` is an unsigned integral type.

**Preconditions**

- ``0 <= rank && rank < num_bits(T)``.

**Performance considerations**

- ``log2(num_bits(T))`` binary-search steps, each of them executing population count and 6 ALU instructions.
- For a non-zero ``value``, ``bit_fns(value, 0)`` is equal to ``cuda::std::countr_zero(value)``.
- For a ``value`` with all bits set, ``bit_fns(value, num_bits(T) - 1)`` is equal to ``num_bits(T) - 1``.

.. note::

    The caller can skip the early return check if the rank is known to be less than the number of set bits in ``value`` by providing the assumption ``rank < cuda::std::popcount(value)`` with ``__builtin_assume`` before the call. A false assumption results in undefined behavior.

Example
-------

.. code:: cuda

    #include <cuda/bit>
    #include <cuda/std/cassert>
    #include <cuda/std/cstdint>
    #include <cuda_runtime_api.h>

    __global__ void bit_fns_kernel() {
        // 0b10110100 has set bits at positions 2, 4, 5, and 7
        assert(cuda::bit_fns(uint32_t{0b10110100}, 0) == 2);
        assert(cuda::bit_fns(uint32_t{0b10110100}, 1) == 4);
        assert(cuda::bit_fns(uint32_t{0b10110100}, 2) == 5);
        assert(cuda::bit_fns(uint32_t{0b10110100}, 3) == 7);
        // with every bit set, rank k is at position k
        assert(cuda::bit_fns(~uint32_t{0}, 31) == 31);
        // there is no set bit of rank 4
        assert(cuda::bit_fns(uint32_t{0b10110100}, 4) == -1);
    }

    int main() {
        bit_fns_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
        return 0;
    }
