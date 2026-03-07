.. _libcudacxx-extended-api-synchronization-pipeline:

``cuda::pipeline``
==================

.. toctree::
   :hidden:
   :maxdepth: 1

   pipeline/role
   pipeline/shared_state
   pipeline/destructor
   pipeline/make_pipeline
   pipeline/quit
   pipeline/consumer_release
   pipeline/consumer_wait
   pipeline/consumer_wait_prior
   pipeline/pipeline_producer_commit
   pipeline/producer_acquire
   pipeline/producer_commit

Defined in header ``<cuda/pipeline>``:

.. code:: cuda

   template <cuda::thread_scope Scope>
   class cuda::pipeline {
   public:
     pipeline() = delete;

     __host__ __device__ ~pipeline();

     pipeline& operator=(pipeline const&) = delete;

     __host__ __device__ void producer_acquire();

     __host__ __device__ void producer_commit();

     __host__ __device__ void consumer_wait();

     template <typename Rep, typename Period>
     __host__ __device__ bool consumer_wait_for(cuda::std::chrono::duration<Rep, Period> const& duration);

     template <typename Clock, typename Duration>
     __host__ __device__
     bool consumer_wait_until(cuda::std::chrono::time_point<Clock, Duration> const& time_point);

     __host__ __device__ void consumer_release();

     __host__ __device__ bool quit();
   };

The class template ``cuda::pipeline`` provides a coordination mechanism which can sequence
:ref:`asynchronous operations <libcudacxx-extended-api-asynchronous-operations>`, such as
:ref:`cuda::memcpy_async <libcudacxx-extended-api-asynchronous-operations-memcpy-async>`, into stages.

A thread interacts with a *pipeline stage* using the following pattern:

  1. Acquire the pipeline stage.
  2. Commit some operations to the stage.
  3. Wait for the previously committed operations to complete.
  4. Release the pipeline stage.

For :ref:`cuda::thread_scope <libcudacxx-extended-api-memory-model-thread-scopes>` ``s`` other than
``cuda::thread_scope_thread``, a
:ref:`cuda::pipeline_shared_state <libcudacxx-extended-api-synchronization-pipeline-pipeline-shared-state>` is
required to coordinate the participating threads.

*Pipelines* can be either *unified* or *partitioned*. In a *unified pipeline*, all the participating threads are both
producers and consumers. In a *partitioned pipeline*, each participating thread is either a producer or a consumer.

.. rubric:: Template Parameters

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - ``Scope``
     - The scope of threads participating in the *pipeline*.

.. rubric:: Member Functions

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Member function
     - Description
   * - (constructor) [deleted]
     - ``cuda::pipeline`` is not constructible.
   * - ``operator=`` [deleted]
     - ``cuda::pipeline`` is not assignable.
   * - :ref:`(destructor) <libcudacxx-extended-api-synchronization-pipeline-pipeline-destructor>`
     -  Destroys the ``cuda::pipeline``.
   * - :ref:`producer_acquire <libcudacxx-extended-api-synchronization-pipeline-pipeline-producer-acquire>`
     - Blocks the current thread until the next *pipeline stage* is available.
   * - :ref:`producer_commit <libcudacxx-extended-api-synchronization-pipeline-pipeline-producer-commit>`
     - Commits operations previously issued by the current thread to the current *pipeline stage*.
   * - :ref:`consumer_wait <libcudacxx-extended-api-synchronization-pipeline-pipeline-consumer-wait>`
     - Blocks the current thread until all operations committed to the current *pipeline stage* complete.
   * - :ref:`consumer_wait_for <libcudacxx-extended-api-synchronization-pipeline-pipeline-consumer-wait>`
     - Blocks the current thread until all operations committed to the current *pipeline stage* complete or after the
       specified timeout duration.
   * - :ref:`consumer_wait_until <libcudacxx-extended-api-synchronization-pipeline-pipeline-consumer-wait>`
     - Blocks the current thread until all operations committed to the current *pipeline stage* complete or until
       specified time point has been reached.
   * - :ref:`consumer_release <libcudacxx-extended-api-synchronization-pipeline-pipeline-consumer-release>`
     - Release the current *pipeline stage*.
   * - :ref:`quit <libcudacxx-extended-api-synchronization-pipeline-pipeline-quit>`
     - Quits current thread's participation in the *pipeline*.

.. note::

   - A thread role cannot change during the lifetime of the pipeline object.

.. rubric:: Example

.. code:: cuda

   #include <cuda/pipeline>
   #include <cooperative_groups.h>

   // Disables `pipeline_shared_state` initialization warning.
   #pragma nv_diag_suppress static_var_with_dynamic_init

   template <typename T>
   __device__ void compute(T* ptr);

   template <typename T>
   __global__ void example_kernel(T* global0, T* global1, cuda::std::size_t subset_count) {
     extern __shared__ T s[];
     auto group = cooperative_groups::this_thread_block();
     T* shared[2] = { s, s + 2 * group.size() };

     // Create a pipeline.
     constexpr auto scope = cuda::thread_scope_block;
     constexpr auto stages_count = 2;
     __shared__ cuda::pipeline_shared_state<scope, stages_count> shared_state;
     auto pipeline = cuda::make_pipeline(group, &shared_state);

     // Prime the pipeline.
     pipeline.producer_acquire();
     cuda::memcpy_async(group, shared[0],
                        &global0[0], sizeof(T) * group.size(), pipeline);
     cuda::memcpy_async(group, shared[0] + group.size(),
                        &global1[0], sizeof(T) * group.size(), pipeline);
     pipeline.producer_commit();

     // Pipelined copy/compute.
     for (cuda::std::size_t subset = 1; subset < subset_count; ++subset) {
       pipeline.producer_acquire();
       cuda::memcpy_async(group, shared[subset % 2],
                          &global0[subset * group.size()],
                          sizeof(T) * group.size(), pipeline);
       cuda::memcpy_async(group, shared[subset % 2] + group.size(),
                          &global1[subset * group.size()],
                          sizeof(T) * group.size(), pipeline);
       pipeline.producer_commit();
       pipeline.consumer_wait();
       compute(shared[(subset - 1) % 2]);
       pipeline.consumer_release();
     }

     // Drain the pipeline.
     pipeline.consumer_wait();
     compute(shared[(subset_count - 1) % 2]);
     pipeline.consumer_release();
   }

   template void __global__ example_kernel<int>(int*, int*, cuda::std::size_t);

`See it on Godbolt <https://godbolt.org/z/zc41bWvja>`_
