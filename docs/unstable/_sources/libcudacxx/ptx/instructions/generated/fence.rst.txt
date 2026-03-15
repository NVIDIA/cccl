..
   This file was automatically generated. Do not edit.

fence.sc.cta
^^^^^^^^^^^^
.. code-block:: cuda

   // fence.sem.scope; // 1. PTX ISA 60, SM_70
   // .sem       = { .sc }
   // .scope     = { .cta, .gpu, .sys }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline void fence(
     cuda::ptx::sem_sc_t,
     cuda::ptx::scope_t<Scope> scope);

fence.sc.gpu
^^^^^^^^^^^^
.. code-block:: cuda

   // fence.sem.scope; // 1. PTX ISA 60, SM_70
   // .sem       = { .sc }
   // .scope     = { .cta, .gpu, .sys }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline void fence(
     cuda::ptx::sem_sc_t,
     cuda::ptx::scope_t<Scope> scope);

fence.sc.sys
^^^^^^^^^^^^
.. code-block:: cuda

   // fence.sem.scope; // 1. PTX ISA 60, SM_70
   // .sem       = { .sc }
   // .scope     = { .cta, .gpu, .sys }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline void fence(
     cuda::ptx::sem_sc_t,
     cuda::ptx::scope_t<Scope> scope);

fence.sc.cluster
^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fence.sem.scope; // 2. PTX ISA 78, SM_90
   // .sem       = { .sc }
   // .scope     = { .cluster }
   template <typename = void>
   __device__ static inline void fence(
     cuda::ptx::sem_sc_t,
     cuda::ptx::scope_cluster_t);

fence.acq_rel.cta
^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fence.sem.scope; // 1. PTX ISA 60, SM_70
   // .sem       = { .acq_rel }
   // .scope     = { .cta, .gpu, .sys }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline void fence(
     cuda::ptx::sem_acq_rel_t,
     cuda::ptx::scope_t<Scope> scope);

fence.acq_rel.gpu
^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fence.sem.scope; // 1. PTX ISA 60, SM_70
   // .sem       = { .acq_rel }
   // .scope     = { .cta, .gpu, .sys }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline void fence(
     cuda::ptx::sem_acq_rel_t,
     cuda::ptx::scope_t<Scope> scope);

fence.acq_rel.sys
^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fence.sem.scope; // 1. PTX ISA 60, SM_70
   // .sem       = { .acq_rel }
   // .scope     = { .cta, .gpu, .sys }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline void fence(
     cuda::ptx::sem_acq_rel_t,
     cuda::ptx::scope_t<Scope> scope);

fence.acq_rel.cluster
^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fence.sem.scope; // 2. PTX ISA 78, SM_90
   // .sem       = { .acq_rel }
   // .scope     = { .cluster }
   template <typename = void>
   __device__ static inline void fence(
     cuda::ptx::sem_acq_rel_t,
     cuda::ptx::scope_cluster_t);

fence.acquire.cta
^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fence.sem.scope; // PTX ISA 86, SM_90
   // .sem       = { .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline void fence(
     cuda::ptx::sem_acquire_t,
     cuda::ptx::scope_t<Scope> scope);

fence.acquire.cluster
^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fence.sem.scope; // PTX ISA 86, SM_90
   // .sem       = { .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline void fence(
     cuda::ptx::sem_acquire_t,
     cuda::ptx::scope_t<Scope> scope);

fence.acquire.gpu
^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fence.sem.scope; // PTX ISA 86, SM_90
   // .sem       = { .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline void fence(
     cuda::ptx::sem_acquire_t,
     cuda::ptx::scope_t<Scope> scope);

fence.acquire.sys
^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fence.sem.scope; // PTX ISA 86, SM_90
   // .sem       = { .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline void fence(
     cuda::ptx::sem_acquire_t,
     cuda::ptx::scope_t<Scope> scope);

fence.release.cta
^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fence.sem.scope; // PTX ISA 86, SM_90
   // .sem       = { .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline void fence(
     cuda::ptx::sem_release_t,
     cuda::ptx::scope_t<Scope> scope);

fence.release.cluster
^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fence.sem.scope; // PTX ISA 86, SM_90
   // .sem       = { .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline void fence(
     cuda::ptx::sem_release_t,
     cuda::ptx::scope_t<Scope> scope);

fence.release.gpu
^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fence.sem.scope; // PTX ISA 86, SM_90
   // .sem       = { .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline void fence(
     cuda::ptx::sem_release_t,
     cuda::ptx::scope_t<Scope> scope);

fence.release.sys
^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fence.sem.scope; // PTX ISA 86, SM_90
   // .sem       = { .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline void fence(
     cuda::ptx::sem_release_t,
     cuda::ptx::scope_t<Scope> scope);
