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

**Return value**

- The zero-based position of the set bit with rank ``rank``, or ``-1`` (``0xFFFFFFFF``, the not-found result of CUDA's ``__fns`` intrinsic) if ``value`` has fewer than ``rank + 1`` set bits.

**Constraints**

- ``T`` is an unsigned integral type.

**Preconditions**

- ``0 <= rank && rank < num_bits(T)``.

**Performance considerations**

The function performs essentially the following operations in device code, for ``T`` up to 64 bits:

- ``POPC``, ``ISETP`` and a branch for the not-found early exit.
- ``log2(num_bits(T))`` binary-search steps, each ``POPC``, ``LOP3``, ``ISETP``, ``SEL`` x2, an integer subtract and ``SHF``, plus the adds that accumulate the position.

.. note::

    If the caller guarantees ``rank < cuda::std::popcount(value)``, using ``__builtin_assume`` before the call can eliminate the not-found check. A false assumption results in undefined behavior.

.. note::

    For a non-zero ``value``, ``bit_fns(value, 0)`` is equal to ``cuda::std::countr_zero(value)``.

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
