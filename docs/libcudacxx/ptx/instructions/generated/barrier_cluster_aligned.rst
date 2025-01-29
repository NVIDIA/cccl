..
   This file was automatically generated. Do not edit.

barrier.cluster.arrive.aligned
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // barrier.cluster.arrive.aligned; // PTX ISA 78, SM_90
   // .aligned   = { .aligned }
   // Marked volatile and as clobbering memory
   template <typename = void>
   __device__ static inline void barrier_cluster_arrive(
     cuda::ptx::dot_aligned_t);

barrier.cluster.wait.aligned
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // barrier.cluster.wait.aligned; // PTX ISA 78, SM_90
   // .aligned   = { .aligned }
   // Marked volatile and as clobbering memory
   template <typename = void>
   __device__ static inline void barrier_cluster_wait(
     cuda::ptx::dot_aligned_t);

barrier.cluster.arrive.release.aligned
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // barrier.cluster.arrive.sem.aligned; // PTX ISA 80, SM_90
   // .sem       = { .release }
   // .aligned   = { .aligned }
   // Marked volatile and as clobbering memory
   template <typename = void>
   __device__ static inline void barrier_cluster_arrive(
     cuda::ptx::sem_release_t,
     cuda::ptx::dot_aligned_t);

barrier.cluster.arrive.relaxed.aligned
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // barrier.cluster.arrive.sem.aligned; // PTX ISA 80, SM_90
   // .sem       = { .relaxed }
   // .aligned   = { .aligned }
   // Marked volatile
   template <typename = void>
   __device__ static inline void barrier_cluster_arrive(
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::dot_aligned_t);

barrier.cluster.wait.acquire.aligned
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // barrier.cluster.wait.sem.aligned; // PTX ISA 80, SM_90
   // .sem       = { .acquire }
   // .aligned   = { .aligned }
   // Marked volatile and as clobbering memory
   template <typename = void>
   __device__ static inline void barrier_cluster_wait(
     cuda::ptx::sem_acquire_t,
     cuda::ptx::dot_aligned_t);
