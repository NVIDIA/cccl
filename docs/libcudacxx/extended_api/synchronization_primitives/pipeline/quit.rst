.. _libcudacxx-extended-api-synchronization-pipeline-pipeline-quit:

cuda::pipeline::quit
========================

Defined in header ``<cuda/pipeline>``:

.. code:: cuda

   template <cuda::thread_scope Scope>
   __host__ __device__
   bool cuda::pipeline<Scope>::quit();

Quits the current thread's participation in the collective ownership of the corresponding
:ref:`cuda::pipeline_shared_state <libcudacxx-extended-api-synchronization-pipeline-pipeline-shared-state>`.
Ownership of :ref:`cuda::pipeline_shared_state <libcudacxx-extended-api-synchronization-pipeline-pipeline-shared-state>`
is released by the last invoking thread.

.. rubric:: Return Value

``true`` if ownership of the *shared state* was released, otherwise ``false``.

.. note::

   - After the completion of a call to ``cuda::pipeline::quit``, no other operations other than
     :ref:`cuda::pipeline::~pipeline <libcudacxx-extended-api-synchronization-pipeline-pipeline-destructor>` may
     called by the current thread.
