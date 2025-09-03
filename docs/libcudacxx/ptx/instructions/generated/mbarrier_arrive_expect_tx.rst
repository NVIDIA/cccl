..
   This file was automatically generated. Do not edit.

mbarrier.arrive.expect_tx.release.cta.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.arrive.expect_tx.sem.scope.space.b64 state, [addr], tx_count; // 8.  PTX ISA 80, SM_90
   // .sem       = { .release }
   // .scope     = { .cta, .cluster }
   // .space     = { .shared::cta }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline uint64_t mbarrier_arrive_expect_tx(
     cuda::ptx::sem_release_t,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::space_shared_t,
     uint64_t* addr,
     const uint32_t& tx_count);

mbarrier.arrive.expect_tx.release.cluster.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.arrive.expect_tx.sem.scope.space.b64 state, [addr], tx_count; // 8.  PTX ISA 80, SM_90
   // .sem       = { .release }
   // .scope     = { .cta, .cluster }
   // .space     = { .shared::cta }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline uint64_t mbarrier_arrive_expect_tx(
     cuda::ptx::sem_release_t,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::space_shared_t,
     uint64_t* addr,
     const uint32_t& tx_count);

mbarrier.arrive.expect_tx.release.cluster.shared::cluster.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.arrive.expect_tx.sem.scope.space.b64   _, [addr], tx_count; // 9.  PTX ISA 80, SM_90
   // .sem       = { .release }
   // .scope     = { .cluster }
   // .space     = { .shared::cluster }
   template <typename = void>
   __device__ static inline void mbarrier_arrive_expect_tx(
     cuda::ptx::sem_release_t,
     cuda::ptx::scope_cluster_t,
     cuda::ptx::space_cluster_t,
     uint64_t* addr,
     const uint32_t& tx_count);

mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.arrive.expect_tx.sem.scope.space.b64 state, [addr], txCount; // PTX ISA 86, SM_90
   // .sem       = { .relaxed }
   // .scope     = { .cta, .cluster }
   // .space     = { .shared::cta }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline uint64_t mbarrier_arrive_expect_tx(
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::space_shared_t,
     uint64_t* addr,
     const uint32_t& txCount);

mbarrier.arrive.expect_tx.relaxed.cluster.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.arrive.expect_tx.sem.scope.space.b64 state, [addr], txCount; // PTX ISA 86, SM_90
   // .sem       = { .relaxed }
   // .scope     = { .cta, .cluster }
   // .space     = { .shared::cta }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline uint64_t mbarrier_arrive_expect_tx(
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::space_shared_t,
     uint64_t* addr,
     const uint32_t& txCount);

mbarrier.arrive.expect_tx.relaxed.cluster.shared::cluster.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.arrive.expect_tx.sem.scope.space.b64 _, [addr], txCount; // PTX ISA 86, SM_90
   // .sem       = { .relaxed }
   // .scope     = { .cluster }
   // .space     = { .shared::cluster }
   template <typename = void>
   __device__ static inline void mbarrier_arrive_expect_tx(
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_cluster_t,
     cuda::ptx::space_cluster_t,
     uint64_t* addr,
     const uint32_t& txCount);
