..
   This file was automatically generated. Do not edit.

fence.mbarrier_init.release.cluster
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fence.mbarrier_init.sem.scope; // 3. PTX ISA 80, SM_90
   // .sem       = { .release }
   // .scope     = { .cluster }
   template <typename = void>
   __device__ static inline void fence_mbarrier_init(
     cuda::ptx::sem_release_t,
     cuda::ptx::scope_cluster_t);
