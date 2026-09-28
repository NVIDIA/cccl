.. _libcudacxx-extended-api-mdspan-copy:

Asynchronous device ``mdspan`` copy
===================================

Defined in the ``<cuda/mdspan>`` header.

.. code:: cuda

    namespace cuda {

    template <class TpIn,
              class ExtentsIn,
              class LayoutPolicyIn,
              class AccessorPolicyIn,
              class TpOut,
              class ExtentsOut,
              class LayoutPolicyOut,
              class AccessorPolicyOut>
    void copy(
        cuda::device_mdspan<TpIn,  ExtentsIn,  LayoutPolicyIn,  AccessorPolicyIn>  src,
        cuda::device_mdspan<TpOut, ExtentsOut, LayoutPolicyOut, AccessorPolicyOut> dst,
        cuda::stream_ref                                                           stream);

    } // namespace cuda

``cuda::copy`` copies elements between two device mdspans. The operation is enqueued on ``stream`` and the function returns without waiting for the copy to finish. Operations submitted earlier to the same stream happen before the copy.
The caller must synchronize the stream, wait on a subsequently recorded event, or establish equivalent stream ordering before accessing the destination or releasing storage referenced by either mdspan.

Template Parameters
-------------------

- ``TpIn``: The source element type.
- ``ExtentsIn``: The source extents type.
- ``LayoutPolicyIn``: The source layout policy.
- ``AccessorPolicyIn``: The source accessor policy wrapped by ``cuda::device_accessor``.
- ``TpOut``: The destination element type.
- ``ExtentsOut``: The destination extents type.
- ``LayoutPolicyOut``: The destination layout policy.
- ``AccessorPolicyOut``: The destination accessor policy wrapped by  ``cuda::device_accessor``.

Parameters
----------

- ``src``: The source device mdspan.
- ``dst``: The destination device mdspan.
- ``stream``: The stream on which the copy is enqueued.

Constraints
-----------

- ``TpIn`` is convertible to ``TpOut``.
- ``TpOut`` is not ``const``.
- ``LayoutPolicyIn`` and ``LayoutPolicyOut`` are each one of ``cuda::std::layout_left``,
  ``cuda::std::layout_right``, ``cuda::std::layout_stride``, or ``cuda::layout_stride_relaxed``.

Preconditions
-------------

- The source and destination have equal total sizes and equal extents after singleton dimensions are removed.
- The destination has no broadcast dimensions and does not have interleaved stride order.
- The source and destination memory regions do not overlap.
- Non-empty mdspans have non-null, sufficiently aligned data handles that remain valid until the stream completes the operation.

Runtime errors
--------------

The function throws ``std::invalid_argument`` when a run-time precondition is violated. CUDA launch and driver failures are reported through the usual CCCL CUDA error handling.

Example
-------

.. code:: cuda

   #include <cuda/mdspan>
   #include <cuda/stream>

   using extents_t = cuda::std::extents<int, 37, 53>;

   cuda::device_mdspan<float, extents_t, cuda::std::layout_right> src{source_ptr};
   cuda::device_mdspan<float, extents_t, cuda::std::layout_left> dst{destination_ptr};
   cuda::stream stream{cuda::device_ref{0}};

   cuda::copy(src, dst, stream);

   // Other host work may execute while the copy is running.
   stream.sync();
