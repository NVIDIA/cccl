..
   This file was automatically generated. Do not edit.

mbarrier.test_wait.parity.shared.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.test_wait.parity.shared.b64 waitComplete, [addr], phaseParity; // PTX ISA 71, SM_80
   template <typename = void>
   __device__ static inline bool mbarrier_test_wait_parity(
     uint64_t* addr,
     const uint32_t& phaseParity);

mbarrier.test_wait.parity.acquire.cta.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.test_wait.parity.sem.scope.shared::cta.b64 waitComplete, [addr], phaseParity; // PTX ISA 80, SM_90
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

   // mbarrier.test_wait.parity.sem.scope.shared::cta.b64 waitComplete, [addr], phaseParity; // PTX ISA 80, SM_90
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

mbarrier.test_wait.parity.phase_type::primary.acquire.cta.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.test_wait.parity.phase_type.sem.scope.shared::cta.b64 waitComplete|isReportSeen, [addr], phaseParity; // PTX ISA 94, SM_90
   // .phase_type = { .phase_type::primary }
   // .sem       = { .acquire, .relaxed }
   // .scope     = { .cta, .cluster }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline bool mbarrier_test_wait_parity(
     cuda::ptx::mbarrier_phase_primary_t,
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     bool& isReportSeen,
     uint64_t* addr,
     uint32_t phaseParity);

mbarrier.test_wait.parity.phase_type::primary.acquire.cluster.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.test_wait.parity.phase_type.sem.scope.shared::cta.b64 waitComplete|isReportSeen, [addr], phaseParity; // PTX ISA 94, SM_90
   // .phase_type = { .phase_type::primary }
   // .sem       = { .acquire, .relaxed }
   // .scope     = { .cta, .cluster }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline bool mbarrier_test_wait_parity(
     cuda::ptx::mbarrier_phase_primary_t,
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     bool& isReportSeen,
     uint64_t* addr,
     uint32_t phaseParity);

mbarrier.test_wait.parity.phase_type::primary.relaxed.cta.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.test_wait.parity.phase_type.sem.scope.shared::cta.b64 waitComplete|isReportSeen, [addr], phaseParity; // PTX ISA 94, SM_90
   // .phase_type = { .phase_type::primary }
   // .sem       = { .acquire, .relaxed }
   // .scope     = { .cta, .cluster }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline bool mbarrier_test_wait_parity(
     cuda::ptx::mbarrier_phase_primary_t,
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     bool& isReportSeen,
     uint64_t* addr,
     uint32_t phaseParity);

mbarrier.test_wait.parity.phase_type::primary.relaxed.cluster.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.test_wait.parity.phase_type.sem.scope.shared::cta.b64 waitComplete|isReportSeen, [addr], phaseParity; // PTX ISA 94, SM_90
   // .phase_type = { .phase_type::primary }
   // .sem       = { .acquire, .relaxed }
   // .scope     = { .cta, .cluster }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline bool mbarrier_test_wait_parity(
     cuda::ptx::mbarrier_phase_primary_t,
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     bool& isReportSeen,
     uint64_t* addr,
     uint32_t phaseParity);


mbarrier.test_wait.parity.phase_type::primary.acquire.cta.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.test_wait.parity.phase_type.sem.scope.shared::cta.b64 waitComplete|isReportSeen, reportValue, [addr], phaseParity; // PTX ISA 94, SM_90
   // .phase_type = { .phase_type::primary }
   // .sem       = { .acquire, .relaxed }
   // .scope     = { .cta, .cluster }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline bool mbarrier_test_wait_parity(
     cuda::ptx::mbarrier_phase_primary_t,
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     bool& isReportSeen,
     uint8_t& reportValue,
     uint64_t* addr,
     uint32_t phaseParity);

mbarrier.test_wait.parity.phase_type::primary.acquire.cluster.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.test_wait.parity.phase_type.sem.scope.shared::cta.b64 waitComplete|isReportSeen, reportValue, [addr], phaseParity; // PTX ISA 94, SM_90
   // .phase_type = { .phase_type::primary }
   // .sem       = { .acquire, .relaxed }
   // .scope     = { .cta, .cluster }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline bool mbarrier_test_wait_parity(
     cuda::ptx::mbarrier_phase_primary_t,
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     bool& isReportSeen,
     uint8_t& reportValue,
     uint64_t* addr,
     uint32_t phaseParity);

mbarrier.test_wait.parity.phase_type::primary.relaxed.cta.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.test_wait.parity.phase_type.sem.scope.shared::cta.b64 waitComplete|isReportSeen, reportValue, [addr], phaseParity; // PTX ISA 94, SM_90
   // .phase_type = { .phase_type::primary }
   // .sem       = { .acquire, .relaxed }
   // .scope     = { .cta, .cluster }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline bool mbarrier_test_wait_parity(
     cuda::ptx::mbarrier_phase_primary_t,
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     bool& isReportSeen,
     uint8_t& reportValue,
     uint64_t* addr,
     uint32_t phaseParity);

mbarrier.test_wait.parity.phase_type::primary.relaxed.cluster.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.test_wait.parity.phase_type.sem.scope.shared::cta.b64 waitComplete|isReportSeen, reportValue, [addr], phaseParity; // PTX ISA 94, SM_90
   // .phase_type = { .phase_type::primary }
   // .sem       = { .acquire, .relaxed }
   // .scope     = { .cta, .cluster }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline bool mbarrier_test_wait_parity(
     cuda::ptx::mbarrier_phase_primary_t,
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     bool& isReportSeen,
     uint8_t& reportValue,
     uint64_t* addr,
     uint32_t phaseParity);

mbarrier.test_wait.parity.phase_type::conditional.acquire.cta.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.test_wait.parity.phase_type.sem.scope.shared::cta.b64 waitComplete, [addr], phaseParity; // PTX ISA 94, SM_90
   // .phase_type = { .phase_type::conditional }
   // .sem       = { .acquire, .relaxed }
   // .scope     = { .cta, .cluster }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline bool mbarrier_test_wait_parity(
     cuda::ptx::mbarrier_phase_conditional_t,
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     uint64_t* addr,
     uint32_t phaseParity);

mbarrier.test_wait.parity.phase_type::conditional.acquire.cluster.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.test_wait.parity.phase_type.sem.scope.shared::cta.b64 waitComplete, [addr], phaseParity; // PTX ISA 94, SM_90
   // .phase_type = { .phase_type::conditional }
   // .sem       = { .acquire, .relaxed }
   // .scope     = { .cta, .cluster }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline bool mbarrier_test_wait_parity(
     cuda::ptx::mbarrier_phase_conditional_t,
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     uint64_t* addr,
     uint32_t phaseParity);

mbarrier.test_wait.parity.phase_type::conditional.relaxed.cta.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.test_wait.parity.phase_type.sem.scope.shared::cta.b64 waitComplete, [addr], phaseParity; // PTX ISA 94, SM_90
   // .phase_type = { .phase_type::conditional }
   // .sem       = { .acquire, .relaxed }
   // .scope     = { .cta, .cluster }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline bool mbarrier_test_wait_parity(
     cuda::ptx::mbarrier_phase_conditional_t,
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     uint64_t* addr,
     uint32_t phaseParity);

mbarrier.test_wait.parity.phase_type::conditional.relaxed.cluster.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.test_wait.parity.phase_type.sem.scope.shared::cta.b64 waitComplete, [addr], phaseParity; // PTX ISA 94, SM_90
   // .phase_type = { .phase_type::conditional }
   // .sem       = { .acquire, .relaxed }
   // .scope     = { .cta, .cluster }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline bool mbarrier_test_wait_parity(
     cuda::ptx::mbarrier_phase_conditional_t,
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     uint64_t* addr,
     uint32_t phaseParity);
