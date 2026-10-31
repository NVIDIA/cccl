..
   This file was automatically generated. Do not edit.

mbarrier.complete_tx.relaxed.cta.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.complete_tx.sem.scope.space.b64 [addr], txCount; // PTX ISA 80, SM_90
   // .sem       = { .relaxed }
   // .scope     = { .cta, .cluster }
   // .space     = { .shared::cta }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline void mbarrier_complete_tx(
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::space_shared_t,
     uint64_t* addr,
     uint32_t txCount);

mbarrier.complete_tx.relaxed.cluster.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.complete_tx.sem.scope.space.b64 [addr], txCount; // PTX ISA 80, SM_90
   // .sem       = { .relaxed }
   // .scope     = { .cta, .cluster }
   // .space     = { .shared::cta }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline void mbarrier_complete_tx(
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::space_shared_t,
     uint64_t* addr,
     uint32_t txCount);

mbarrier.complete_tx.relaxed.cta.shared::cluster.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.complete_tx.sem.scope.space.b64 [addr], txCount; // PTX ISA 80, SM_90
   // .sem       = { .relaxed }
   // .scope     = { .cta, .cluster }
   // .space     = { .shared::cluster }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline void mbarrier_complete_tx(
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::space_cluster_t,
     uint64_t* addr,
     uint32_t txCount);

mbarrier.complete_tx.relaxed.cluster.shared::cluster.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.complete_tx.sem.scope.space.b64 [addr], txCount; // PTX ISA 80, SM_90
   // .sem       = { .relaxed }
   // .scope     = { .cta, .cluster }
   // .space     = { .shared::cluster }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline void mbarrier_complete_tx(
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::space_cluster_t,
     uint64_t* addr,
     uint32_t txCount);

mbarrier.complete_tx.relaxed.cta.shared::cluster.multicast::cluster::32b.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.complete_tx.sem.scope.space.multicast::cluster::32b.b64 [addr], txCount, ctaMask; // PTX ISA 94, SM_107a, SM_107f
   // .sem       = { .relaxed }
   // .scope     = { .cta, .cluster }
   // .space     = { .shared::cluster }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline void mbarrier_complete_tx_multicast_32b(
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::space_cluster_t,
     uint64_t* addr,
     uint32_t txCount,
     uint32_t ctaMask);

mbarrier.complete_tx.relaxed.cluster.shared::cluster.multicast::cluster::32b.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.complete_tx.sem.scope.space.multicast::cluster::32b.b64 [addr], txCount, ctaMask; // PTX ISA 94, SM_107a, SM_107f
   // .sem       = { .relaxed }
   // .scope     = { .cta, .cluster }
   // .space     = { .shared::cluster }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline void mbarrier_complete_tx_multicast_32b(
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::space_cluster_t,
     uint64_t* addr,
     uint32_t txCount,
     uint32_t ctaMask);
