.. _libcudacxx-extended-api-synchronization-pipeline-pipeline-role:

cuda::pipeline_role
=======================

Defined in header ``<cuda/pipeline>``:

.. code:: cuda

   enum class pipeline_role : /* unspecified */ {
     producer,
     consumer
   };

``cuda::pipeline_role`` specifies the role of a particular thread in a partitioned producer/consumer pipeline.

Constants
---------

| ``producer`` \| A producer thread that generates data and issuing
  :ref:`asynchronous operations <libcudacxx-extended-api-asynchronous-operations>`. \|
| ``consumer`` \| A consumer thread that consumes data and waiting for previously
  :ref:`asynchronous operations <libcudacxx-extended-api-asynchronous-operations>` to complete. \|

Example
-------

See the :ref:`cuda::make_pipeline example <libcudacxx-extended-api-synchronization-pipeline-pipeline-make-pipeline-example>`.
