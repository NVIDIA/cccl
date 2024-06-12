.. _libcudacxx-extended-api-synchronization-pipeline-pipeline-producer-commit:

cuda::pipeline::producer_commit
===================================

Defined in header ``<cuda/pipeline>``:

.. code:: cuda

   template <cuda::thread_scope Scope>
   __host__ __device__
   void cuda::pipeline<Scope>::producer_commit();

Commits operations previously issued by the current thread to the current *pipeline stage*.

.. note::

   - If the calling thread is a *consumer thread*, the behavior is undefined.
   - If the pipeline is in a :ref:`quitted state <libcudacxx-extended-api-synchronization-pipeline-pipeline-quit>`,
     the behavior is undefined.
