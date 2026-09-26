..
   This file was automatically generated. Do not edit.

mbarrier.test_wait.shared.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.test_wait.shared.b64 waitComplete, [addr], state; // PTX ISA 70, SM_80
   template <typename = void>
   __device__ static inline bool mbarrier_test_wait(
     uint64_t* addr,
     const uint64_t& state);

mbarrier.test_wait.acquire.cta.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.test_wait.sem.scope.shared::cta.b64 waitComplete, [addr], state; // PTX ISA 80, SM_90
   // .sem       = { .acquire }
   // .scope     = { .cta, .cluster }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline bool mbarrier_test_wait(
     cuda::ptx::sem_acquire_t,
     cuda::ptx::scope_t<Scope> scope,
     uint64_t* addr,
     const uint64_t& state);

mbarrier.test_wait.acquire.cluster.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.test_wait.sem.scope.shared::cta.b64 waitComplete, [addr], state; // PTX ISA 80, SM_90
   // .sem       = { .acquire }
   // .scope     = { .cta, .cluster }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline bool mbarrier_test_wait(
     cuda::ptx::sem_acquire_t,
     cuda::ptx::scope_t<Scope> scope,
     uint64_t* addr,
     const uint64_t& state);

mbarrier.test_wait.relaxed.cta.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.test_wait.sem.scope.shared::cta.b64 waitComplete, [addr], state; // PTX ISA 86, SM_90
   // .sem       = { .relaxed }
   // .scope     = { .cta, .cluster }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline bool mbarrier_test_wait(
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_t<Scope> scope,
     uint64_t* addr,
     const uint64_t& state);

mbarrier.test_wait.relaxed.cluster.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.test_wait.sem.scope.shared::cta.b64 waitComplete, [addr], state; // PTX ISA 86, SM_90
   // .sem       = { .relaxed }
   // .scope     = { .cta, .cluster }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline bool mbarrier_test_wait(
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_t<Scope> scope,
     uint64_t* addr,
     const uint64_t& state);

mbarrier.test_wait.phase_type::primary.acquire.cta.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.test_wait.phase_type.sem.scope.shared::cta.b64 waitComplete|isReportSeen, [addr], state; // PTX ISA 94, SM_90
   // .phase_type = { .phase_type::primary }
   // .sem       = { .acquire, .relaxed }
   // .scope     = { .cta, .cluster }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline bool mbarrier_test_wait(
     cuda::ptx::mbarrier_phase_primary_t,
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     bool& isReportSeen,
     uint64_t* addr,
     uint64_t state);

mbarrier.test_wait.phase_type::primary.acquire.cluster.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.test_wait.phase_type.sem.scope.shared::cta.b64 waitComplete|isReportSeen, [addr], state; // PTX ISA 94, SM_90
   // .phase_type = { .phase_type::primary }
   // .sem       = { .acquire, .relaxed }
   // .scope     = { .cta, .cluster }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline bool mbarrier_test_wait(
     cuda::ptx::mbarrier_phase_primary_t,
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     bool& isReportSeen,
     uint64_t* addr,
     uint64_t state);

mbarrier.test_wait.phase_type::primary.relaxed.cta.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.test_wait.phase_type.sem.scope.shared::cta.b64 waitComplete|isReportSeen, [addr], state; // PTX ISA 94, SM_90
   // .phase_type = { .phase_type::primary }
   // .sem       = { .acquire, .relaxed }
   // .scope     = { .cta, .cluster }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline bool mbarrier_test_wait(
     cuda::ptx::mbarrier_phase_primary_t,
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     bool& isReportSeen,
     uint64_t* addr,
     uint64_t state);

mbarrier.test_wait.phase_type::primary.relaxed.cluster.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.test_wait.phase_type.sem.scope.shared::cta.b64 waitComplete|isReportSeen, [addr], state; // PTX ISA 94, SM_90
   // .phase_type = { .phase_type::primary }
   // .sem       = { .acquire, .relaxed }
   // .scope     = { .cta, .cluster }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline bool mbarrier_test_wait(
     cuda::ptx::mbarrier_phase_primary_t,
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     bool& isReportSeen,
     uint64_t* addr,
     uint64_t state);

mbarrier.test_wait.phase_type::primary.acquire.cta.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.test_wait.phase_type.sem.scope.shared::cta.b64 waitComplete|isReportSeen, reportValue, [addr], state; // PTX ISA 94, SM_90
   // .phase_type = { .phase_type::primary }
   // .sem       = { .acquire, .relaxed }
   // .scope     = { .cta, .cluster }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline bool mbarrier_test_wait(
     cuda::ptx::mbarrier_phase_primary_t,
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     bool& isReportSeen,
     uint8_t& reportValue,
     uint64_t* addr,
     uint64_t state);

mbarrier.test_wait.phase_type::primary.acquire.cluster.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.test_wait.phase_type.sem.scope.shared::cta.b64 waitComplete|isReportSeen, reportValue, [addr], state; // PTX ISA 94, SM_90
   // .phase_type = { .phase_type::primary }
   // .sem       = { .acquire, .relaxed }
   // .scope     = { .cta, .cluster }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline bool mbarrier_test_wait(
     cuda::ptx::mbarrier_phase_primary_t,
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     bool& isReportSeen,
     uint8_t& reportValue,
     uint64_t* addr,
     uint64_t state);

mbarrier.test_wait.phase_type::primary.relaxed.cta.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.test_wait.phase_type.sem.scope.shared::cta.b64 waitComplete|isReportSeen, reportValue, [addr], state; // PTX ISA 94, SM_90
   // .phase_type = { .phase_type::primary }
   // .sem       = { .acquire, .relaxed }
   // .scope     = { .cta, .cluster }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline bool mbarrier_test_wait(
     cuda::ptx::mbarrier_phase_primary_t,
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     bool& isReportSeen,
     uint8_t& reportValue,
     uint64_t* addr,
     uint64_t state);

mbarrier.test_wait.phase_type::primary.relaxed.cluster.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // mbarrier.test_wait.phase_type.sem.scope.shared::cta.b64 waitComplete|isReportSeen, reportValue, [addr], state; // PTX ISA 94, SM_90
   // .phase_type = { .phase_type::primary }
   // .sem       = { .acquire, .relaxed }
   // .scope     = { .cta, .cluster }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline bool mbarrier_test_wait(
     cuda::ptx::mbarrier_phase_primary_t,
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     bool& isReportSeen,
     uint8_t& reportValue,
     uint64_t* addr,
     uint64_t state);
