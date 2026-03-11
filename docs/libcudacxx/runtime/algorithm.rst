.. _cccl-runtime-algorithm:

Algorithm
==========

The ``runtime`` part of the ``cuda/algorithm`` header provides stream-ordered, byte-wise primitives that operate on ``cuda::std::span`` and
``cuda::std::mdspan``-compatible types. They require a ``cuda::stream_ref`` to enqueue work.

``cuda::copy_bytes``
---------------------
.. _cccl-runtime-algorithm-copy_bytes:

Launch a byte-wise copy from source to destination on the provided stream.

- Signature: ``copy_bytes(stream, src, dst, config = {})``
- Overloads accept ``cuda::std::span``-convertible contiguous ranges or ``cuda::std::mdspan``-convertible multi-dimensional views.
- Elements must be trivially copyable
- ``cuda::std::mdspan``-convertible types must convert to an mdspan that is exhaustive
- The optional ``config`` argument is a ``cuda::copy_configuration`` that controls source access order and managed-memory location hints

Availability: CCCL 3.1.0 / CUDA 13.1

.. code:: cpp

   #include <cuda/algorithm>
   #include <cuda/stream>
   #include <cuda/std/algorithm>
   #include <cuda/std/span>

   void copy_example(cuda::stream_ref s, cuda::std::span<const int> src, cuda::std::span<int> dst) {
     // copy_bytes copies up to src.size_bytes(); dst can be larger.
     auto n = cuda::std::min(src.size(), dst.size());
     auto src_prefix = src.first(n / 2);
     auto dst_prefix = dst.first(n / 2);
     auto src_suffix = src.subspan(n / 2);
     auto dst_suffix = dst.subspan(n / 2);

     // Default behavior: enqueue a stream-ordered byte-wise copy on stream s.
     cuda::copy_bytes(s, src_prefix, dst_prefix);

     // Advanced behavior: customize source access order for this copy.
     auto config = cuda::copy_configuration{
       .src_access_order = cuda::source_access_order::during_api_call,
     };
     cuda::copy_bytes(s, src_suffix, dst_suffix, config);
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
   #include <cuda/std/algorithm>
   #include <cuda/std/span>

   void fill_example(cuda::stream_ref s, cuda::std::span<unsigned char> dst) {
     // Reserve 16-byte red zones at both ends and clear the payload in between.
     auto guard = cuda::std::min(static_cast<decltype(dst.size())>(16), dst.size() / 2);
     auto head  = dst.first(guard);
     auto body  = dst.subspan(guard, dst.size() - 2 * guard);
     auto tail  = dst.last(guard);

     cuda::fill_bytes(s, head, 0xCD); // debug guard pattern
     cuda::fill_bytes(s, body, 0x00); // initialize payload
     cuda::fill_bytes(s, tail, 0xCD); // debug guard pattern
   }
