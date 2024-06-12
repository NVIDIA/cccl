.. _libcudacxx-extended-api-synchronization-pipeline-pipeline-consumer-wait-prior:

cuda::pipeline_consumer_wait_prior
======================================

Defined in header ``<cuda/pipeline>``:

.. code:: cuda

   template <cuda::std::uint8_t Prior>
   __host__ __device__
   void cuda::pipeline_consumer_wait_prior(cuda::pipeline<thread_scope_thread>& pipe);

Let *Stage* be the pipeline stage ``Prior`` stages before the current one (counting the current one).
Blocks the current thread until all operations committed to *pipeline stages* up to *Stage* complete.
All stages up to *Stage* (exclusive) are implicitly released.

.. rubric:: Template Parameters

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - ``Prior``
     - The index of the pipeline stage *Stage* (see above) counting up from the current one. The index of the current stage is ``0``.

.. rubric:: Parameters

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - ``pipe``
     - The thread-scoped ``cuda::pipeline`` object to wait on.

.. note::

   - If the pipeline is in a :ref:`quitted state <libcudacxx-extended-api-synchronization-pipeline-pipeline-quit>`,
     the behavior is undefined.

.. rubric:: Example

.. code:: cuda

   #include <cuda/pipeline>

   __global__ void example_kernel(uint64_t* global, cuda::std::size_t element_count) {
     extern __shared__ uint64_t shared[];

     cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
     for (cuda::std::size_t i = 0; i < element_count; ++i) {
       pipe.producer_acquire();
       cuda::memcpy_async(shared + i, global + i, sizeof(*global), pipe);
       pipe.producer_commit();
     }

     // Wait for operations committed in all stages but the last one.
     cuda::pipeline_consumer_wait_prior<1>(pipe);
     pipe.consumer_release();

     // Wait for operations committed in all stages.
     cuda::pipeline_consumer_wait_prior<0>(pipe);
     pipe.consumer_release();
   }

`See it on Godbolt <https://godbolt.org/z/aT5hb84PY>`_
