.. _libcudacxx-extended-api-memory-ptr_ranges_overlap:

``cuda::ptr_ranges_overlap``
============================

.. code:: cuda

   namespace cuda {

   [[nodiscard]] __host__ __device__ inline
   bool ptr_ranges_overlap(const void* lhs, size_t lhs_count,
                           const void* rhs, size_t rhs_count) noexcept

   } // namespace cuda

Returns ``true`` when the half-open byte ranges ``[lhs, lhs + lhs_count)`` and
``[rhs, rhs + rhs_count)`` intersect.

**Parameters**

- ``lhs``: Pointer to the beginning of the first range.
- ``lhs_count``: Number of bytes in the first range.
- ``rhs``: Pointer to the beginning of the second range.
- ``rhs_count``: Number of bytes in the second range.

**Return value**

- ``true`` when the two ranges overlap, ``false`` otherwise.

**Preconditions**

- ``lhs_count`` and ``rhs_count`` describe valid ranges. In particular, the
  computation of ``lhs + lhs_count`` and ``rhs + rhs_count`` must not overflow.

Example
-------

.. code:: cuda

    #include <cuda/memory>

    struct packet {
        int id;
        float payload[4];
    };

    __host__ __device__ bool buffers_overlap(packet* a, packet* b, size_t count) {
        const size_t bytes = sizeof(packet) * count;
        return cuda::ptr_ranges_overlap(a, bytes, b, bytes);
    }

    int main() {
        packet data[4]{};

        packet* first = &data[0];
        packet* overlap = &data[1];
        packet* disjoint = &data[3];

        bool expect_overlap = buffers_overlap(first, overlap, 2);   // true
        bool expect_disjoint = buffers_overlap(first, disjoint, 1);  // false

        return expect_overlap && !expect_disjoint ? 0 : 1;
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/86zq9Wh55>`_

