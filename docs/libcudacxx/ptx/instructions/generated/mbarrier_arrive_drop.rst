..
   This file was automatically generated. Do not edit.

mbarrier.arrive_drop.release.cta.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.arrive_drop.sem.scope.space.b64 state, [addr], count; // PTX ISA 80, SM_90
   // .sem       = { .release }
   // .scope     = { .cta, .cluster }
   // .space     = { .shared::cta }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline uint64_t mbarrier_arrive_drop(
     cuda::ptx::sem_release_t,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::space_shared_t,
     uint64_t* addr,
     uint32_t count);

mbarrier.arrive_drop.release.cluster.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.arrive_drop.sem.scope.space.b64 state, [addr], count; // PTX ISA 80, SM_90
   // .sem       = { .release }
   // .scope     = { .cta, .cluster }
   // .space     = { .shared::cta }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline uint64_t mbarrier_arrive_drop(
     cuda::ptx::sem_release_t,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::space_shared_t,
     uint64_t* addr,
     uint32_t count);

mbarrier.arrive_drop.release.cluster.shared::cluster.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.arrive_drop.sem.scope.space.b64 _, [addr], count; // PTX ISA 80, SM_90
   // .sem       = { .release }
   // .scope     = { .cluster }
   // .space     = { .shared::cluster }
   template <typename = void>
   __device__ static inline void mbarrier_arrive_drop(
     cuda::ptx::sem_release_t,
     cuda::ptx::scope_cluster_t,
     cuda::ptx::space_cluster_t,
     uint64_t* addr,
     uint32_t count);

mbarrier.arrive_drop.relaxed.cta.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.arrive_drop.sem.scope.space.b64 state, [addr], count; // PTX ISA 86, SM_90
   // .sem       = { .relaxed }
   // .scope     = { .cta, .cluster }
   // .space     = { .shared::cta }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline uint64_t mbarrier_arrive_drop(
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::space_shared_t,
     uint64_t* addr,
     uint32_t count);

mbarrier.arrive_drop.relaxed.cluster.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.arrive_drop.sem.scope.space.b64 state, [addr], count; // PTX ISA 86, SM_90
   // .sem       = { .relaxed }
   // .scope     = { .cta, .cluster }
   // .space     = { .shared::cta }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline uint64_t mbarrier_arrive_drop(
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::space_shared_t,
     uint64_t* addr,
     uint32_t count);

mbarrier.arrive_drop.relaxed.cluster.shared::cluster.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.arrive_drop.sem.scope.space.b64 _, [addr], count; // PTX ISA 86, SM_90
   // .sem       = { .relaxed }
   // .scope     = { .cluster }
   // .space     = { .shared::cluster }
   template <typename = void>
   __device__ static inline void mbarrier_arrive_drop(
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_cluster_t,
     cuda::ptx::space_cluster_t,
     uint64_t* addr,
     uint32_t count);

mbarrier.arrive_drop.release.cluster.shared::cluster.multicast::cluster::32b.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.arrive_drop.sem.scope.space.multicast::cluster::32b.b64 _, [addr], count, ctaMask; // PTX ISA 94, SM_107a, SM_107f
   // .sem       = { .release, .relaxed }
   // .scope     = { .cluster }
   // .space     = { .shared::cluster }
   template <cuda::ptx::dot_sem Sem>
   __device__ static inline void mbarrier_arrive_drop_multicast(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_cluster_t,
     cuda::ptx::space_cluster_t,
     uint64_t* addr,
     uint32_t count,
     uint32_t ctaMask);

mbarrier.arrive_drop.relaxed.cluster.shared::cluster.multicast::cluster::32b.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.arrive_drop.sem.scope.space.multicast::cluster::32b.b64 _, [addr], count, ctaMask; // PTX ISA 94, SM_107a, SM_107f
   // .sem       = { .release, .relaxed }
   // .scope     = { .cluster }
   // .space     = { .shared::cluster }
   template <cuda::ptx::dot_sem Sem>
   __device__ static inline void mbarrier_arrive_drop_multicast(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_cluster_t,
     cuda::ptx::space_cluster_t,
     uint64_t* addr,
     uint32_t count,
     uint32_t ctaMask);

mbarrier.arrive_drop.expect_tx.release.cluster.shared::cluster.multicast::cluster::32b.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.arrive_drop.expect_tx.sem.scope.space.multicast::cluster::32b.b64 _, [addr], tx_count, ctaMask; // PTX ISA 94, SM_107a, SM_107f
   // .sem       = { .release, .relaxed }
   // .scope     = { .cluster }
   // .space     = { .shared::cluster }
   template <cuda::ptx::dot_sem Sem>
   __device__ static inline void mbarrier_arrive_drop_expect_tx_multicast(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_cluster_t,
     cuda::ptx::space_cluster_t,
     uint64_t* addr,
     uint32_t tx_count,
     uint32_t ctaMask);

mbarrier.arrive_drop.expect_tx.relaxed.cluster.shared::cluster.multicast::cluster::32b.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.arrive_drop.expect_tx.sem.scope.space.multicast::cluster::32b.b64 _, [addr], tx_count, ctaMask; // PTX ISA 94, SM_107a, SM_107f
   // .sem       = { .release, .relaxed }
   // .scope     = { .cluster }
   // .space     = { .shared::cluster }
   template <cuda::ptx::dot_sem Sem>
   __device__ static inline void mbarrier_arrive_drop_expect_tx_multicast(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_cluster_t,
     cuda::ptx::space_cluster_t,
     uint64_t* addr,
     uint32_t tx_count,
     uint32_t ctaMask);

mbarrier.arrive_drop.noComplete.release.cta.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.arrive_drop.noComplete.release.cta.shared::cta.b64 state, [addr], count; // PTX ISA 80, SM_80
   template <typename = void>
   __device__ static inline uint64_t mbarrier_arrive_drop_no_complete(
     uint64_t* addr,
     uint32_t count);

mbarrier.arrive_drop.expect_tx.release.cta.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.arrive_drop.expect_tx.sem.scope.space.b64 state, [addr], tx_count; // PTX ISA 80, SM_90
   // .sem       = { .release }
   // .scope     = { .cta, .cluster }
   // .space     = { .shared::cta }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline uint64_t mbarrier_arrive_drop_expect_tx(
     cuda::ptx::sem_release_t,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::space_shared_t,
     uint64_t* addr,
     uint32_t tx_count);

mbarrier.arrive_drop.expect_tx.release.cluster.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.arrive_drop.expect_tx.sem.scope.space.b64 state, [addr], tx_count; // PTX ISA 80, SM_90
   // .sem       = { .release }
   // .scope     = { .cta, .cluster }
   // .space     = { .shared::cta }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline uint64_t mbarrier_arrive_drop_expect_tx(
     cuda::ptx::sem_release_t,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::space_shared_t,
     uint64_t* addr,
     uint32_t tx_count);

mbarrier.arrive_drop.expect_tx.release.cluster.shared::cluster.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.arrive_drop.expect_tx.sem.scope.space.b64 _, [addr], tx_count; // PTX ISA 80, SM_90
   // .sem       = { .release }
   // .scope     = { .cluster }
   // .space     = { .shared::cluster }
   template <typename = void>
   __device__ static inline void mbarrier_arrive_drop_expect_tx(
     cuda::ptx::sem_release_t,
     cuda::ptx::scope_cluster_t,
     cuda::ptx::space_cluster_t,
     uint64_t* addr,
     uint32_t tx_count);

mbarrier.arrive_drop.expect_tx.relaxed.cta.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.arrive_drop.expect_tx.sem.scope.space.b64 state, [addr], tx_count; // PTX ISA 86, SM_90
   // .sem       = { .relaxed }
   // .scope     = { .cta, .cluster }
   // .space     = { .shared::cta }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline uint64_t mbarrier_arrive_drop_expect_tx(
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::space_shared_t,
     uint64_t* addr,
     uint32_t tx_count);

mbarrier.arrive_drop.expect_tx.relaxed.cluster.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.arrive_drop.expect_tx.sem.scope.space.b64 state, [addr], tx_count; // PTX ISA 86, SM_90
   // .sem       = { .relaxed }
   // .scope     = { .cta, .cluster }
   // .space     = { .shared::cta }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline uint64_t mbarrier_arrive_drop_expect_tx(
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::space_shared_t,
     uint64_t* addr,
     uint32_t tx_count);

mbarrier.arrive_drop.expect_tx.relaxed.cluster.shared::cluster.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.arrive_drop.expect_tx.sem.scope.space.b64 _, [addr], tx_count; // PTX ISA 86, SM_90
   // .sem       = { .relaxed }
   // .scope     = { .cluster }
   // .space     = { .shared::cluster }
   template <typename = void>
   __device__ static inline void mbarrier_arrive_drop_expect_tx(
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_cluster_t,
     cuda::ptx::space_cluster_t,
     uint64_t* addr,
     uint32_t tx_count);
