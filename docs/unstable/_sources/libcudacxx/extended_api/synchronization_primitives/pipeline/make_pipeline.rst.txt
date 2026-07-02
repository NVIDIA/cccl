.. _libcudacxx-extended-api-synchronization-pipeline-pipeline-make-pipeline:

cuda::make_pipeline
=======================

Defined in header ``<cuda/pipeline>``:

.. code:: cuda

   // (1)
   __host__ __device__
   cuda::pipeline<cuda::thread_scope_thread> cuda::make_pipeline();

   // (2)
   template <typename Group,
             cuda::thread_scope Scope,
             cuda::std::uint8_t StagesCount>
   __host__ __device__
   cuda::pipeline<Scope>
   cuda::make_pipeline(Group const& group,
                       cuda::pipeline_shared_state<Scope, StagesCount>* shared_state);

   // (3)
   template <typename Group,
             cuda::thread_scope Scope,
             cuda::std::uint8_t StagesCount>
   __host__ __device__
   cuda::pipeline<Scope>
   cuda::make_pipeline(Group const& group,
                       cuda::pipeline_shared_state<Scope, StagesCount>* shared_state,
                       cuda::std::size_t producer_count);

   // (4)
   template <typename Group,
             cuda::thread_scope Scope,
             cuda::std::uint8_t StagesCount>
   __host__ __device__
   cuda::pipeline<Scope>
   cuda::make_pipeline(Group const& group,
                       cuda::pipeline_shared_state<Scope, StagesCount>* shared_state,
                       cuda::pipeline_role role);

1. Creates a *unified pipeline* such that the calling thread is the only participating thread and performs both
   producer and consumer actions.
2. Creates a *unified pipeline* such that all the threads in ``group`` are performing both producer and consumer actions.
3. Creates a *partitioned pipeline* such that ``producer_threads`` number of threads in ``group`` are performing
   producer actions while the others are performing consumer actions.
4. Creates a *partitioned pipeline* where each thread's role is explicitly specified.

.. rubric:: Template Parameters

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - ``Group``
     - A type satisfying the :ref:`ThreadGroup <libcudacxx-extended-api-thread-groups>` concept.

.. rubric:: Parameters

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - ``group``
     - The group of threads.
   * - ``shared_state``
     - A pointer to an object of type :ref:`cuda::pipeline_shared_state\<Scope\> <libcudacxx-extended-api-synchronization-pipeline-pipeline-shared-state>`
       with ``Scope`` including all the threads in ``group``.
   * - ``producer_count``
     - The number of *producer threads* in the pipeline.
   * - ``role``
     - The role of the current thread in the pipeline.

.. rubric:: Return Value

A ``cuda::pipeline`` object.

.. note::

  - All threads in ``group`` acquire collective ownership of the ``shared_state`` storage.
  - ``make_pipeline`` must be invoked by every threads in ``group`` such that ``group::sync`` may be invoked.
  - ``shared_state`` and ``producer_count`` must be the same across all threads in ``group``, else the behavior is undefined.
  - ``producer_count`` must be strictly inferior to ``group::size``, otherwise the behavior is undefined.

.. _libcudacxx-extended-api-synchronization-pipeline-pipeline-make-pipeline-example:

.. rubric:: Example

.. code:: cuda

   #include <cuda/pipeline>
   #include <cooperative_groups.h>

   // Disables `pipeline_shared_state` initialization warning.
   #pragma nv_diag_suppress static_var_with_dynamic_init

   __global__ void example_kernel() {
     __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> pss0;
     __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> pss1;
     __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> pss2;

     auto group = cooperative_groups::this_thread_block();

     // Create a single thread scoped pipeline.
     cuda::pipeline<cuda::thread_scope_thread> p0 = cuda::make_pipeline();

     // Create a unified block-scoped pipeline.
     cuda::pipeline<cuda::thread_scope_block> p1 = cuda::make_pipeline(group, &pss0);

     // Create a partitioned block-scoped pipeline where half the threads are producers.
     cuda::std::size_t producer_count = group.size() / 2;
     cuda::pipeline<cuda::thread_scope_block> p2
       = cuda::make_pipeline(group, &pss1, producer_count);

     // Create a partitioned block-scoped pipeline where all threads with an even
     // `thread_rank` are producers.
     auto thread_role = (group.thread_rank() % 2)
                        ? cuda::pipeline_role::producer
                       : cuda::pipeline_role::consumer;
     cuda::pipeline<cuda::thread_scope_block> p3
       = cuda::make_pipeline(group, &pss2, thread_role);
   }

`See it on Godbolt <https://godbolt.org/z/aPcGEr64j>`_
