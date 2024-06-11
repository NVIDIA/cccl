.. _libcudacxx-extended-api-synchronization-pipeline-pipeline-consumer-release:

cuda::pipeline::consumer_release
====================================

Defined in header ``<cuda/pipeline>``:

.. code:: cuda

   template <cuda::thread_scope Scope>
   __host__ __device__
   void cuda::pipeline<Scope>::consumer_release();

Releases the current *pipeline stage*.

.. note::

   - If the calling thread is a *producer thread*, the behavior is undefined.
   - If the pipeline is in a :ref:`quitted state <libcudacxx-extended-api-synchronization-pipeline-pipeline-quit>`,
     the behavior is undefined.
