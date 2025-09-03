..
   This file was automatically generated. Do not edit.

mbarrier.try_wait.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.try_wait.shared::cta.b64         waitComplete, [addr], state;                                      // 5a.  PTX ISA 78, SM_90
   template <typename = void>
   __device__ static inline bool mbarrier_try_wait(
     uint64_t* addr,
     const uint64_t& state);

mbarrier.try_wait.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.try_wait.shared::cta.b64         waitComplete, [addr], state, suspendTimeHint;                    // 5b.  PTX ISA 78, SM_90
   template <typename = void>
   __device__ static inline bool mbarrier_try_wait(
     uint64_t* addr,
     const uint64_t& state,
     const uint32_t& suspendTimeHint);

mbarrier.try_wait.acquire.cta.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.try_wait.sem.scope.shared::cta.b64         waitComplete, [addr], state;                        // 6a.  PTX ISA 80, SM_90
   // .sem       = { .acquire }
   // .scope     = { .cta, .cluster }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline bool mbarrier_try_wait(
     cuda::ptx::sem_acquire_t,
     cuda::ptx::scope_t<Scope> scope,
     uint64_t* addr,
     const uint64_t& state);

mbarrier.try_wait.acquire.cluster.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.try_wait.sem.scope.shared::cta.b64         waitComplete, [addr], state;                        // 6a.  PTX ISA 80, SM_90
   // .sem       = { .acquire }
   // .scope     = { .cta, .cluster }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline bool mbarrier_try_wait(
     cuda::ptx::sem_acquire_t,
     cuda::ptx::scope_t<Scope> scope,
     uint64_t* addr,
     const uint64_t& state);

mbarrier.try_wait.acquire.cta.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.try_wait.sem.scope.shared::cta.b64         waitComplete, [addr], state , suspendTimeHint;      // 6b.  PTX ISA 80, SM_90
   // .sem       = { .acquire }
   // .scope     = { .cta, .cluster }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline bool mbarrier_try_wait(
     cuda::ptx::sem_acquire_t,
     cuda::ptx::scope_t<Scope> scope,
     uint64_t* addr,
     const uint64_t& state,
     const uint32_t& suspendTimeHint);

mbarrier.try_wait.acquire.cluster.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.try_wait.sem.scope.shared::cta.b64         waitComplete, [addr], state , suspendTimeHint;      // 6b.  PTX ISA 80, SM_90
   // .sem       = { .acquire }
   // .scope     = { .cta, .cluster }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline bool mbarrier_try_wait(
     cuda::ptx::sem_acquire_t,
     cuda::ptx::scope_t<Scope> scope,
     uint64_t* addr,
     const uint64_t& state,
     const uint32_t& suspendTimeHint);

mbarrier.try_wait.relaxed.cta.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.try_wait.sem.scope.shared::cta.b64 waitComplete, [addr], state, suspendTimeHint; // PTX ISA 86, SM_90
   // .sem       = { .relaxed }
   // .scope     = { .cta, .cluster }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline bool mbarrier_try_wait(
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_t<Scope> scope,
     uint64_t* addr,
     const uint64_t& state,
     const uint32_t& suspendTimeHint);

mbarrier.try_wait.relaxed.cluster.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.try_wait.sem.scope.shared::cta.b64 waitComplete, [addr], state, suspendTimeHint; // PTX ISA 86, SM_90
   // .sem       = { .relaxed }
   // .scope     = { .cta, .cluster }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline bool mbarrier_try_wait(
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_t<Scope> scope,
     uint64_t* addr,
     const uint64_t& state,
     const uint32_t& suspendTimeHint);

mbarrier.try_wait.relaxed.cta.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.try_wait.sem.scope.shared::cta.b64 waitComplete, [addr], state; // PTX ISA 86, SM_90
   // .sem       = { .relaxed }
   // .scope     = { .cta, .cluster }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline bool mbarrier_try_wait(
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_t<Scope> scope,
     uint64_t* addr,
     const uint64_t& state);

mbarrier.try_wait.relaxed.cluster.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.try_wait.sem.scope.shared::cta.b64 waitComplete, [addr], state; // PTX ISA 86, SM_90
   // .sem       = { .relaxed }
   // .scope     = { .cta, .cluster }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline bool mbarrier_try_wait(
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_t<Scope> scope,
     uint64_t* addr,
     const uint64_t& state);
