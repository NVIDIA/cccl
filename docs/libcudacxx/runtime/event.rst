.. _cccl-runtime-event:

Events
======

Event is a snapshot of execution state of a stream. It can be used to synchronize work submitted to a stream up to a certain point, establish dependency between streams or measure time passed between two events.

``cuda::event_ref``
--------------------------------------------------
.. _cccl-runtime-event-event-ref:

``cuda::event_ref`` is a non-owning wrapper around a ``cudaEvent_t``. It prevents unsafe implicit constructions from
``nullptr`` or integer literals and provides convenient helpers:

- ``record(cuda::stream_ref)``: record the event on a stream
- ``sync()``: wait for the recorded work to complete
- ``is_done()``: non-blocking completion query
- comparison operators against other ``event_ref`` or ``cudaEvent_t``

Availability: CCCL 3.1.0 / CUDA 13.1

Example:

.. code:: cpp

   #include <cuda/stream>

   void record_on_stream(cuda::stream_ref stream, cudaEvent_t raw_handle) {
     cuda::event_ref e{raw_handle};
     e.record(stream);
   }

``cuda::event``
--------------------------------------------
.. _cccl-runtime-event-event:

``cuda::event`` is an owning wrapper around a ``cudaEvent_t`` (with timing disabled). It inherits from ``event_ref`` and provides all of its functionality.
It also creates and destroys the native event, can be moved (but not copied), and can release ownership via ``release()``. Construction can target a specific
``cuda::device_ref`` or record immediately on a ``cuda::stream_ref``.

Availability: CCCL 3.1.0 / CUDA 13.1

.. code:: cpp

   #include <cuda/stream>
   #include <cuda/devices>
   #include <cuda/std/optional>

   cuda::std::optional<cuda::event> query_and_record_on_stream(cuda::stream_ref stream) {
     if (stream.is_done()) {
       return cuda::std::nullopt;
     } else {
       return cuda::event{stream};
     }
   }

.. _cccl-runtime-event-timed-event:

``cuda::timed_event``
-----------------------------------------------------

``cuda::timed_event`` is an owning wrapper for a timed ``cudaEvent_t``. It inherits from ``event`` and provides all of its functionality.
It also supports elapsed-time queries between two events via ``operator-``, returning ``cuda::std::chrono::nanoseconds``.

Availability: CCCL 3.1.0 / CUDA 13.1

.. code:: cpp

   #include <cuda/stream>
   #include <cuda/std/chrono>

   template <typename F>
   cuda::std::chrono::nanoseconds measure_execution_time(cuda::stream_ref stream, F&& f) {
     cuda::timed_event start{stream};
     f(stream);
     cuda::timed_event end{stream};
     return end - start;
   }
