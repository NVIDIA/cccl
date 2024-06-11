.. _libcudacxx-ptx-instructions-mbarrier-test_wait:

mbarrier.test_wait
==================

-  PTX ISA:
   `mbarrier.test_wait <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-mbarrier-try-wait>`__

.. _mbarrier.test_wait-1:

mbarrier.test_wait
------------------

mbarrier.test_wait.shared.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // mbarrier.test_wait.shared.b64 waitComplete, [addr], state;                                                  // 1.  PTX ISA 70, SM_80
   template <typename=void>
   __device__ static inline bool mbarrier_test_wait(
     uint64_t* addr,
     const uint64_t& state);

mbarrier.test_wait.acquire.cta.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // mbarrier.test_wait{.sem}{.scope}.shared::cta.b64        waitComplete, [addr], state;                        // 2.   PTX ISA 80, SM_90
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
.. code:: cuda

   // mbarrier.test_wait{.sem}{.scope}.shared::cta.b64        waitComplete, [addr], state;                        // 2.   PTX ISA 80, SM_90
   // .sem       = { .acquire }
   // .scope     = { .cta, .cluster }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline bool mbarrier_test_wait(
     cuda::ptx::sem_acquire_t,
     cuda::ptx::scope_t<Scope> scope,
     uint64_t* addr,
     const uint64_t& state);

mbarrier.test_wait.parity
-------------------------

mbarrier.test_wait.parity.shared.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // mbarrier.test_wait.parity.shared.b64 waitComplete, [addr], phaseParity;                                     // 3.  PTX ISA 71, SM_80
   template <typename=void>
   __device__ static inline bool mbarrier_test_wait_parity(
     uint64_t* addr,
     const uint32_t& phaseParity);

mbarrier.test_wait.parity.acquire.cta.shared::cta.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // mbarrier.test_wait.parity{.sem}{.scope}.shared::cta.b64 waitComplete, [addr], phaseParity;                  // 4.  PTX ISA 80, SM_90
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
.. code:: cuda

   // mbarrier.test_wait.parity{.sem}{.scope}.shared::cta.b64 waitComplete, [addr], phaseParity;                  // 4.  PTX ISA 80, SM_90
   // .sem       = { .acquire }
   // .scope     = { .cta, .cluster }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline bool mbarrier_test_wait_parity(
     cuda::ptx::sem_acquire_t,
     cuda::ptx::scope_t<Scope> scope,
     uint64_t* addr,
     const uint32_t& phaseParity);
