.. _cccl-runtime-stream:

Streams
=======

Stream is conceptually a queue of operations for a specific device. It is passed as an argument to all asynchronous operations like kernel launch, memory copy and allocations.

``cuda::stream_ref``
---------------------
.. _cccl-runtime-stream-stream-ref:

``cuda::stream_ref`` is a non-owning wrapper around a ``cudaStream_t``. It prevents unsafe implicit constructions from
``nullptr`` or integer literals and provides convenient helpers for:

- ``sync()``: wait for the recorded work to complete
- ``is_done()``: non-blocking completion query
- comparison operators against other ``stream_ref`` or ``cudaStream_t``

Availability: CCCL 2.2.0 / CUDA 12.3

Example:

.. code:: cpp

    #include <cuda/stream>

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cuda::stream_ref ref{stream};

    ref.sync();            // synchronizes the stream via cudaStreamSynchronize
    assert(ref.is_done()); // verifies that the stream has finished all operations via cudaStreamQuery

    // compare against other stream_ref or cudaStream_t
    assert(ref == stream);
    assert(ref != cuda::invalid_stream);

    cudaStreamDestroy(stream);

``cuda::stream``
-----------------
.. _cccl-runtime-stream-stream:

``cuda::stream`` is an owning wrapper around a ``cudaStream_t`` that manages the lifetime of the underlying CUDA stream.
It derives from ``stream_ref``, provides all of its functionality, and can be used anywhere a ``stream_ref`` is expected.
It can be constructed for a specific ``cuda::device_ref``, moved (but not copied), and converted from or to a
``cudaStream_t`` via ``from_native_handle``/``release()``.

Availability: CCCL 3.1.0 / CUDA 13.1

.. code:: cpp

   #include <cuda/stream>
   #include <cuda/devices>

   int main() {
     {
       // Create a stream on a specific device
       cuda::stream s{cuda::devices[0]};

      // Pass to a stream-ordered API

       // Synchronize the stream
       s.sync();
     } // Stream is automatically destroyed here
   }
