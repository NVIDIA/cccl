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

- ``next_stream()``: returns the next stream in round-robin order
- ``at(i)`` and ``operator[](i)``: return the stream in slot ``i % size()``
- ``size()``, ``device()``, ``priority()``: the parameters given at construction

The constructor throws ``std::invalid_argument`` if the pool has a size of zero.

Every stream of the pool is created like a :cpp:struct:`cuda::stream`: non-blocking with respect to the legacy
default stream, with the priority given at construction. These are the only creation parameters
:cpp:struct:`cuda::stream` exposes, and the pool forwards them unchanged.

Both getters return a :cpp:class:`cuda::stream_ref` that stays valid for the lifetime of the pool. The streams are
destroyed with the pool, so the work submitted to them must be synchronized before the pool goes away; the pool does
not do it. When the streams are created is chosen with a ``cuda::stream_pool_creation`` value passed after the size:

- ``stream_pool_creation::eager``, the default: every stream is created in the constructor.
- ``stream_pool_creation::lazy``: a stream is created by the first request for its slot. Two threads racing for the
  same empty slot both create a stream; one publishes it and the other destroys its own.

The getters can be called concurrently from several threads. The pool takes no lock: its synchronization is
lock-free, but not wait-free, since the round-robin advance retries a compare-exchange that only fails when another
caller succeeded. The first request for an empty slot of a lazy pool pays the cost of the stream creation, a driver
call.

A pool cannot be copied or moved. Code that needs to hand a pool around, store it in a container, or share it
between several owners should allocate it with ``std::make_unique`` or ``std::make_shared``.

Availability: CCCL 3.6.0

.. code:: cpp

   #include <cuda/stream>
   #include <cuda/devices>

   int main() {
     // 16 streams on device 0, all created here
     cuda::stream_pool pool{cuda::devices[0], 16};

     for (int i = 0; i < 64; ++i) {
       // Cycles through the 16 streams
       cuda::stream_ref s = pool.next_stream();
       // Pass to a stream-ordered API
     }

     // Always the same stream, for work that must stay ordered
     cuda::stream_ref fixed = pool[3];

     // Synchronize streams with submitted work before the pool is destroyed
     for (std::size_t i = 0; i < pool.size(); ++i) {
       pool[i].sync();
     }
   } // All streams are destroyed here
