..
   This file was automatically generated. Do not edit.

fence.proxy.async::generic.acquire.sync_restrict::shared::cluster.cluster
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // fence.proxy.async::generic.sem.sync_restrict::space.scope; // PTX ISA 86, SM_90
   // .sem       = { .acquire }
   // .space     = { .shared::cluster }
   // .scope     = { .cluster }
   template <typename = void>
   __device__ static inline void fence_proxy_async_generic_sync_restrict(
     cuda::ptx::sem_acquire_t,
     cuda::ptx::space_cluster_t,
     cuda::ptx::scope_cluster_t);

fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // fence.proxy.async::generic.sem.sync_restrict::space.scope; // PTX ISA 86, SM_90
   // .sem       = { .release }
   // .space     = { .shared::cta }
   // .scope     = { .cluster }
   template <typename = void>
   __device__ static inline void fence_proxy_async_generic_sync_restrict(
     cuda::ptx::sem_release_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::scope_cluster_t);
