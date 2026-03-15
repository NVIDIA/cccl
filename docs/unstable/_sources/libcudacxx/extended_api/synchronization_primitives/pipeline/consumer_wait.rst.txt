.. _libcudacxx-extended-api-synchronization-pipeline-pipeline-consumer-wait:

cuda::pipeline::consumer_wait
=================================

Defined in header ``<cuda/pipeline>``:

.. code:: cuda

   // (1)
   template <cuda::thread_scope Scope>
   __host__ __device__
   void cuda::pipeline<Scope>::consumer_wait();

   // (2)
   template <cuda::thread_scope Scope>
   template <typename Rep, typename Period>
   __host__ __device__
   bool cuda::pipeline<Scope>::consumer_wait_for(
     cuda::std::chrono::duration<Rep, Period> const& duration);

   // (3)
   template <cuda::thread_scope Scope>
   template <typename Clock, typename Duration>
   __host__ __device__
   bool cuda::pipeline<Scope>::consumer_wait_until(
     cuda::std::chrono::time_point<Clock, Duration> const& time_point);

1. Blocks the current thread until all operations committed to the current *pipeline stage* complete.
2. Blocks the current thread until all operations committed to the current *pipeline stage* complete or after the
   specified timeout duration.
3. Blocks the current thread until all operations committed to the current *pipeline stage* complete or until specified
   time point has been reached.

.. rubric:: Parameters

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - ``duration``
     - An object of type ``cuda::std::chrono::duration`` representing the maximum time to spend waiting.
   * - ``time_point``
     - An object of type ``cuda::std::chrono::time_point`` representing the time when to stop waiting.


.. rubric:: Return Value

``false`` if the *wait* timed out, ``true`` otherwise.

.. note::

   - If the calling thread is a *producer thread*, the behavior is undefined.
   - If the pipeline is in a :ref:`quitted state <libcudacxx-extended-api-synchronization-pipeline-pipeline-quit>`,
     the behavior is undefined.
