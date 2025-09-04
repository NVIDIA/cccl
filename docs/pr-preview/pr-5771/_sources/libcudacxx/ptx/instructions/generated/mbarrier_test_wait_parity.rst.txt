..
   This file was automatically generated. Do not edit.

mbarrier.test_wait.parity.shared.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.test_wait.parity.shared.b64 waitComplete, [addr], phaseParity;                                     // 3.  PTX ISA 71, SM_80
   template <typename = void>
   __device__ static inline bool mbarrier_test_wait_parity(
     uint64_t* addr,
     const uint32_t& phaseParity);

mbarrier.test_wait.parity.acquire.cta.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.test_wait.parity.sem.scope.shared::cta.b64 waitComplete, [addr], phaseParity;                  // 4.  PTX ISA 80, SM_90
   // .sem       = { .acquire }
   // .scope     = { .cta, .cluster }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline bool mbarrier_test_wait_parity(
     cuda::ptx::sem_acquire_t,
     cuda::ptx::scope_t<Scope> scope,
     uint64_t* addr,
     const uint32_t& phaseParity);

mbarrier.test_wait.parity.acquire.cluster.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.test_wait.parity.sem.scope.shared::cta.b64 waitComplete, [addr], phaseParity;                  // 4.  PTX ISA 80, SM_90
   // .sem       = { .acquire }
   // .scope     = { .cta, .cluster }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline bool mbarrier_test_wait_parity(
     cuda::ptx::sem_acquire_t,
     cuda::ptx::scope_t<Scope> scope,
     uint64_t* addr,
     const uint32_t& phaseParity);

mbarrier.test_wait.parity.relaxed.cta.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.test_wait.parity.sem.scope.shared::cta.b64 waitComplete, [addr], phaseParity; // PTX ISA 86, SM_90
   // .sem       = { .relaxed }
   // .scope     = { .cta, .cluster }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline bool mbarrier_test_wait_parity(
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_t<Scope> scope,
     uint64_t* addr,
     const uint32_t& phaseParity);

mbarrier.test_wait.parity.relaxed.cluster.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.test_wait.parity.sem.scope.shared::cta.b64 waitComplete, [addr], phaseParity; // PTX ISA 86, SM_90
   // .sem       = { .relaxed }
   // .scope     = { .cta, .cluster }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline bool mbarrier_test_wait_parity(
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_t<Scope> scope,
     uint64_t* addr,
     const uint32_t& phaseParity);
