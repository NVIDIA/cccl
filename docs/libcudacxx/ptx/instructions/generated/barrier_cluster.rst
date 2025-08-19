..
   This file was automatically generated. Do not edit.

barrier.cluster.arrive
^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // barrier.cluster.arrive; // PTX ISA 78, SM_90
   // Marked volatile and as clobbering memory
   template <typename = void>
   __device__ static inline void barrier_cluster_arrive();

barrier.cluster.wait
^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // barrier.cluster.wait; // PTX ISA 78, SM_90
   // Marked volatile and as clobbering memory
   template <typename = void>
   __device__ static inline void barrier_cluster_wait();

barrier.cluster.arrive.release
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // barrier.cluster.arrive.sem; // PTX ISA 80, SM_90
   // .sem       = { .release }
   // Marked volatile and as clobbering memory
   template <typename = void>
   __device__ static inline void barrier_cluster_arrive(
     cuda::ptx::sem_release_t);

barrier.cluster.arrive.relaxed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // barrier.cluster.arrive.sem; // PTX ISA 80, SM_90
   // .sem       = { .relaxed }
   // Marked volatile
   template <typename = void>
   __device__ static inline void barrier_cluster_arrive(
     cuda::ptx::sem_relaxed_t);

barrier.cluster.wait.acquire
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // barrier.cluster.wait.sem; // PTX ISA 80, SM_90
   // .sem       = { .acquire }
   // Marked volatile and as clobbering memory
   template <typename = void>
   __device__ static inline void barrier_cluster_wait(
     cuda::ptx::sem_acquire_t);
