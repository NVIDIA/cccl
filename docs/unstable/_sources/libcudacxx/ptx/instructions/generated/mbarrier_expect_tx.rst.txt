..
   This file was automatically generated. Do not edit.

mbarrier.expect_tx.relaxed.cta.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.expect_tx.sem.scope.space.b64 [addr], txCount; // 1. PTX ISA 80, SM_90
   // .sem       = { .relaxed }
   // .scope     = { .cta, .cluster }
   // .space     = { .shared::cta }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline void mbarrier_expect_tx(
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::space_shared_t,
     uint64_t* addr,
     uint32_t txCount);

mbarrier.expect_tx.relaxed.cluster.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.expect_tx.sem.scope.space.b64 [addr], txCount; // 1. PTX ISA 80, SM_90
   // .sem       = { .relaxed }
   // .scope     = { .cta, .cluster }
   // .space     = { .shared::cta }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline void mbarrier_expect_tx(
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::space_shared_t,
     uint64_t* addr,
     uint32_t txCount);

mbarrier.expect_tx.relaxed.cta.shared::cluster.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.expect_tx.sem.scope.space.b64 [addr], txCount; // 2. PTX ISA 80, SM_90
   // .sem       = { .relaxed }
   // .scope     = { .cta, .cluster }
   // .space     = { .shared::cluster }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline void mbarrier_expect_tx(
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::space_cluster_t,
     uint64_t* addr,
     uint32_t txCount);

mbarrier.expect_tx.relaxed.cluster.shared::cluster.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.expect_tx.sem.scope.space.b64 [addr], txCount; // 2. PTX ISA 80, SM_90
   // .sem       = { .relaxed }
   // .scope     = { .cta, .cluster }
   // .space     = { .shared::cluster }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline void mbarrier_expect_tx(
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::space_cluster_t,
     uint64_t* addr,
     uint32_t txCount);
