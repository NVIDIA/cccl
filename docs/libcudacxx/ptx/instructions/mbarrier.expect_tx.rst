.. _libcudacxx-ptx-instructions-mbarrier-expect_tx:

mbarrier.expect_tx
==================

-  PTX ISA:
   `mbarrier.expect_tx <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-expect-tx>`__

mbarrier.expect_tx.relaxed.cta.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

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
.. code:: cuda

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
.. code:: cuda

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
.. code:: cuda

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
