.. _libcudacxx-extended-api-synchronization-pipeline-pipeline-producer-acquire:

cuda::pipeline::producer_acquire
====================================

Defined in header ``<cuda/pipeline>``:

.. code:: cuda

   template <cuda::thread_scope Scope>
   __host__ __device__
   void cuda::pipeline<Scope>::producer_acquire();

Blocks the current thread until the next *pipeline stage* is available.

.. note::

   - If the calling thread is a *consumer thread*, the behavior is undefined.
   - If the pipeline is in a :ref:`quitted state <libcudacxx-extended-api-synchronization-pipeline-pipeline-quit>`,
     the behavior is undefined.
