.. _cccl-runtime-stream:

Streams
========

Stream is conceptually a queue of operations for a specific device. It is passed as an argument to all asynchronous operations like kernel launch, memory copy and allocations.

:cpp:class:`cuda::stream_ref`
-------------------------------
.. _cccl-runtime-stream-stream-ref:

:cpp:class:`cuda::stream_ref` is a non-owning wrapper around a ``cudaStream_t``. It prevents unsafe implicit constructions from
``nullptr`` or integer literals and provides convenient helpers for:

- ``sync()``: wait for the recorded work to complete
- ``is_done()``: non-blocking completion query
- comparison operators against other :cpp:class:`cuda::stream_ref` or ``cudaStream_t``

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

:cpp:struct:`cuda::stream`
---------------------------
.. _cccl-runtime-stream-stream:

:cpp:struct:`cuda::stream` is an owning wrapper around a ``cudaStream_t`` that manages the lifetime of the underlying CUDA
stream.
It derives from :cpp:class:`cuda::stream_ref`, provides all of its functionality, and can be used anywhere a
:cpp:class:`cuda::stream_ref` is expected.
It can be constructed for a specific :cpp:class:`cuda::device_ref`, moved (but not copied), and converted from or to a
``cudaStream_t`` via ``from_native_handle``/``release()``.
Constructing a new :cpp:struct:`cuda::stream` always creates a non-blocking stream; see
:ref:`non-blocking stream creation <cccl-runtime-cudart-non-blocking-streams>` for CUDA Runtime interop details.

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

:cpp:class:`cuda::stream_pool`
-------------------------------
.. _cccl-runtime-stream-stream-pool:

:cpp:class:`cuda::stream_pool` owns a fixed number of non-blocking :cpp:struct:`cuda::stream` objects created on one
device or green context. It is meant for code that wants to spread independent work over a few streams without
managing their lifetime.

- ``get_stream()``: returns the next stream in round-robin order
- ``get_stream(i)``: returns the stream in slot ``i % size()``
- ``streams()``: returns all ``size()`` streams, creating the ones not handed out yet
- ``size()``, ``device()``, ``priority()``: the parameters given at construction

Both getters return a :cpp:class:`cuda::stream_ref` that stays valid for the lifetime of the pool, including across a
move of the pool. Streams are created on first use and destroyed with the pool. A pool can be moved but not copied,
and all getters can be called concurrently from several threads.

Availability: CCCL 3.6.0

.. code:: cpp

   #include <cuda/stream>
   #include <cuda/devices>

   int main() {
     // 16 streams on device 0, none created yet
     cuda::stream_pool pool{cuda::devices[0]};

     for (int i = 0; i < 64; ++i) {
       // Cycles through the 16 streams, creating each one the first time it is handed out
       cuda::stream_ref s = pool.get_stream();
       // Pass to a stream-ordered API
     }

     // Always the same stream, for work that must stay ordered
     cuda::stream_ref fixed = pool.get_stream(3);

     // Wait for everything submitted to the pool
     for (cuda::stream_ref s : pool.streams()) {
       s.sync();
     }
   } // All streams are destroyed here
