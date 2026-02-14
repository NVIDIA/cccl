.. _cccl-runtime-algorithm:

Algorithm
==========

The ``runtime`` part of the ``cuda/algorithm`` header provides stream-ordered, byte-wise primitives that operate on ``cuda::std::span`` and
``cuda::std::mdspan``-compatible types. They require a ``cuda::stream_ref`` to enqueue work.

``cuda::copy_bytes``
---------------------
.. _cccl-runtime-algorithm-copy_bytes:

Launch a byte-wise copy from source to destination on the provided stream.

- Overloads accept ``cuda::std::span``-convertible contiguous ranges or ``cuda::std::mdspan``-convertible multi-dimensional views.
- Elements must be trivially copyable
- ``cuda::std::mdspan``-convertible types must convert to an mdspan that is exhaustive
- Source access order (during the copy call or in stream order) can be configured with ``cuda::copy_configuration``

Availability: CCCL 3.1.0 / CUDA 13.1

.. code:: cpp

   #include <cuda/algorithm>
   #include <cuda/stream>
   #include <cuda/std/span>

   void copy_example(cuda::stream_ref s, int* d_dst, const int* d_src, std::size_t n) {
     cuda::std::span<const int> src{d_src, n};
     cuda::std::span<int>       dst{d_dst, n};
     cuda::copy_bytes(s, src, dst);  // enqueued on s
   }


``cuda::fill_bytes``
---------------------
.. _cccl-runtime-algorithm-fill_bytes:

Launch a byte-wise fill of the destination on the provided stream.

- Overloads accept ``cuda::std::span``-convertible or ``cuda::std::mdspan``-convertible destinations.
- Elements must be trivially copyable
- ``cuda::std::mdspan``-convertible types must convert to an mdspan that is exhaustive

Availability: CCCL 3.1.0 / CUDA 13.1

.. code:: cpp

   #include <cuda/algorithm>
   #include <cuda/stream>
   #include <cuda/std/span>

   void fill_example(cuda::stream_ref s, int* d_dst, std::size_t n) {
     cuda::std::span<int> dst{d_dst, n};
     cuda::fill_bytes(s, dst, 0x00); // zero-fill device memory
   }
