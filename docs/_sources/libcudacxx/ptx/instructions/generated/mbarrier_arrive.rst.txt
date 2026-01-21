..
   This file was automatically generated. Do not edit.

mbarrier.arrive.shared.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.arrive.shared.b64                                  state,  [addr];           // 1.  PTX ISA 70, SM_80
   template <typename = void>
   __device__ static inline uint64_t mbarrier_arrive(
     uint64_t* addr);

mbarrier.arrive.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.arrive.shared::cta.b64                             state,  [addr], count;    // 2.  PTX ISA 78, SM_90
   template <typename = void>
   __device__ static inline uint64_t mbarrier_arrive(
     uint64_t* addr,
     const uint32_t& count);

mbarrier.arrive.release.cta.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.arrive.sem.scope.space.b64                   state,  [addr];           // 3a.  PTX ISA 80, SM_90
   // .sem       = { .release }
   // .scope     = { .cta, .cluster }
   // .space     = { .shared::cta }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline uint64_t mbarrier_arrive(
     cuda::ptx::sem_release_t,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::space_shared_t,
     uint64_t* addr);

mbarrier.arrive.release.cluster.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.arrive.sem.scope.space.b64                   state,  [addr];           // 3a.  PTX ISA 80, SM_90
   // .sem       = { .release }
   // .scope     = { .cta, .cluster }
   // .space     = { .shared::cta }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline uint64_t mbarrier_arrive(
     cuda::ptx::sem_release_t,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::space_shared_t,
     uint64_t* addr);

mbarrier.arrive.release.cta.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.arrive.sem.scope.space.b64                   state,  [addr], count;    // 3b.  PTX ISA 80, SM_90
   // .sem       = { .release }
   // .scope     = { .cta, .cluster }
   // .space     = { .shared::cta }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline uint64_t mbarrier_arrive(
     cuda::ptx::sem_release_t,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::space_shared_t,
     uint64_t* addr,
     const uint32_t& count);

mbarrier.arrive.release.cluster.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.arrive.sem.scope.space.b64                   state,  [addr], count;    // 3b.  PTX ISA 80, SM_90
   // .sem       = { .release }
   // .scope     = { .cta, .cluster }
   // .space     = { .shared::cta }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline uint64_t mbarrier_arrive(
     cuda::ptx::sem_release_t,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::space_shared_t,
     uint64_t* addr,
     const uint32_t& count);

mbarrier.arrive.release.cluster.shared::cluster.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.arrive.sem.scope.space.b64                   _, [addr];                // 4a.  PTX ISA 80, SM_90
   // .sem       = { .release }
   // .scope     = { .cluster }
   // .space     = { .shared::cluster }
   template <typename = void>
   __device__ static inline void mbarrier_arrive(
     cuda::ptx::sem_release_t,
     cuda::ptx::scope_cluster_t,
     cuda::ptx::space_cluster_t,
     uint64_t* addr);

mbarrier.arrive.release.cluster.shared::cluster.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.arrive.sem.scope.space.b64                   _, [addr], count;         // 4b.  PTX ISA 80, SM_90
   // .sem       = { .release }
   // .scope     = { .cluster }
   // .space     = { .shared::cluster }
   template <typename = void>
   __device__ static inline void mbarrier_arrive(
     cuda::ptx::sem_release_t,
     cuda::ptx::scope_cluster_t,
     cuda::ptx::space_cluster_t,
     uint64_t* addr,
     const uint32_t& count);

mbarrier.arrive.relaxed.cta.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.arrive.sem.scope.space.b64 state, [addr], count; // PTX ISA 86, SM_90
   // .sem       = { .relaxed }
   // .scope     = { .cta, .cluster }
   // .space     = { .shared::cta }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline uint64_t mbarrier_arrive(
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::space_shared_t,
     uint64_t* addr,
     const uint32_t& count);

mbarrier.arrive.relaxed.cluster.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.arrive.sem.scope.space.b64 state, [addr], count; // PTX ISA 86, SM_90
   // .sem       = { .relaxed }
   // .scope     = { .cta, .cluster }
   // .space     = { .shared::cta }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline uint64_t mbarrier_arrive(
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::space_shared_t,
     uint64_t* addr,
     const uint32_t& count);

mbarrier.arrive.relaxed.cta.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.arrive.sem.scope.space.b64 state, [addr]; // PTX ISA 86, SM_90
   // .sem       = { .relaxed }
   // .scope     = { .cta, .cluster }
   // .space     = { .shared::cta }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline uint64_t mbarrier_arrive(
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::space_shared_t,
     uint64_t* addr);

mbarrier.arrive.relaxed.cluster.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.arrive.sem.scope.space.b64 state, [addr]; // PTX ISA 86, SM_90
   // .sem       = { .relaxed }
   // .scope     = { .cta, .cluster }
   // .space     = { .shared::cta }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline uint64_t mbarrier_arrive(
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::space_shared_t,
     uint64_t* addr);

mbarrier.arrive.relaxed.cluster.shared::cluster.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.arrive.sem.scope.space.b64 _, [addr], count; // PTX ISA 86, SM_90
   // .sem       = { .relaxed }
   // .scope     = { .cluster }
   // .space     = { .shared::cluster }
   template <typename = void>
   __device__ static inline void mbarrier_arrive(
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_cluster_t,
     cuda::ptx::space_cluster_t,
     uint64_t* addr,
     const uint32_t& count);

mbarrier.arrive.relaxed.cluster.shared::cluster.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.arrive.sem.scope.space.b64 _, [addr]; // PTX ISA 86, SM_90
   // .sem       = { .relaxed }
   // .scope     = { .cluster }
   // .space     = { .shared::cluster }
   template <typename = void>
   __device__ static inline void mbarrier_arrive(
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_cluster_t,
     cuda::ptx::space_cluster_t,
     uint64_t* addr);
