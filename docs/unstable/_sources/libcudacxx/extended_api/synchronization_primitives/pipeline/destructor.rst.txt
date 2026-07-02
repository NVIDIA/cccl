.. _libcudacxx-extended-api-synchronization-pipeline-pipeline-destructor:

cuda::pipeline::~pipeline
=============================

Defined in header ``<cuda/pipeline>``:

.. code:: cuda

   template <cuda::thread_scope Scope>
   __host__ __device__
   cuda::pipeline<Scope>::~pipeline();

Destructs the pipeline. Calls :ref:`cuda::pipeline::quit <libcudacxx-extended-api-synchronization-pipeline-pipeline-quit>`
if it was not called by the current thread and destructs the pipeline.
