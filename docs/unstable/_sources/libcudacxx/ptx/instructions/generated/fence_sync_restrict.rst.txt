..
   This file was automatically generated. Do not edit.

fence.acquire.sync_restrict::shared::cluster.cluster
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fence.sem.sync_restrict::space.scope; // PTX ISA 86, SM_90
   // .sem       = { .acquire }
   // .space     = { .shared::cluster }
   // .scope     = { .cluster }
   template <typename = void>
   __device__ static inline void fence_sync_restrict(
     cuda::ptx::sem_acquire_t,
     cuda::ptx::space_cluster_t,
     cuda::ptx::scope_cluster_t);

fence.release.sync_restrict::shared::cta.cluster
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fence.sem.sync_restrict::space.scope; // PTX ISA 86, SM_90
   // .sem       = { .release }
   // .space     = { .shared::cta }
   // .scope     = { .cluster }
   template <typename = void>
   __device__ static inline void fence_sync_restrict(
     cuda::ptx::sem_release_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::scope_cluster_t);
