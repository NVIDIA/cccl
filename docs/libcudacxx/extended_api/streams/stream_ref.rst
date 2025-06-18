.. _libcudacxx-extended-api-streams-stream-ref:

``cuda::stream_ref``
====================

CUDA `stream-ordered allocations <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-ordered-memory-allocator>`__
rely on ``cudaStream_t`` as a handle to the cuda stream.

However, as this is just an alias for a plain pointer type it carries with it a lot of common pitfals around implicit
conversions from e.g ``nullptr`` or a literal ``0``.

These hard to spot bugs can be avoided through ``cuda::stream_ref``, which is a simple wrapper around a ``cudaStream_t``
that prevents implicit conversions. It also provides the ``wait()`` and ``ready()`` member functions to facilitate
waiting for a stream to finish and checking whether it is finished.

.. code:: cpp

       cudaStream_t stream;
       cudaStreamCreate(&stream);
       cuda::stream_ref ref{stream};

       ref.wait();          // synchronizes the stream via cudaStreamSynchronize
       assert(ref.ready()); // verifies that the stream has finished all operations via cudaStreamQuery
       cudaStreamDestroy(stream);
