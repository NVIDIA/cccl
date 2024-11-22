fence.sc.cta
^^^^^^^^^^^^
.. code:: cuda

   // fence{.sem}.scope; // 1. PTX ISA 60, SM_70
   // .sem       = { .sc, .acq_rel }
   // .scope     = { .cta, .gpu, .sys }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void fence(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope);

fence.sc.gpu
^^^^^^^^^^^^
.. code:: cuda

   // fence{.sem}.scope; // 1. PTX ISA 60, SM_70
   // .sem       = { .sc, .acq_rel }
   // .scope     = { .cta, .gpu, .sys }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void fence(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope);

fence.sc.sys
^^^^^^^^^^^^
.. code:: cuda

   // fence{.sem}.scope; // 1. PTX ISA 60, SM_70
   // .sem       = { .sc, .acq_rel }
   // .scope     = { .cta, .gpu, .sys }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void fence(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope);

fence.acq_rel.cta
^^^^^^^^^^^^^^^^^
.. code:: cuda

   // fence{.sem}.scope; // 1. PTX ISA 60, SM_70
   // .sem       = { .sc, .acq_rel }
   // .scope     = { .cta, .gpu, .sys }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void fence(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope);

fence.acq_rel.gpu
^^^^^^^^^^^^^^^^^
.. code:: cuda

   // fence{.sem}.scope; // 1. PTX ISA 60, SM_70
   // .sem       = { .sc, .acq_rel }
   // .scope     = { .cta, .gpu, .sys }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void fence(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope);

fence.acq_rel.sys
^^^^^^^^^^^^^^^^^
.. code:: cuda

   // fence{.sem}.scope; // 1. PTX ISA 60, SM_70
   // .sem       = { .sc, .acq_rel }
   // .scope     = { .cta, .gpu, .sys }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void fence(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope);

fence.sc.cluster
^^^^^^^^^^^^^^^^
.. code:: cuda

   // fence{.sem}.scope; // 2. PTX ISA 78, SM_90
   // .sem       = { .sc, .acq_rel }
   // .scope     = { .cluster }
   template <cuda::ptx::dot_sem Sem>
   __device__ static inline void fence(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_cluster_t);

fence.acq_rel.cluster
^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // fence{.sem}.scope; // 2. PTX ISA 78, SM_90
   // .sem       = { .sc, .acq_rel }
   // .scope     = { .cluster }
   template <cuda::ptx::dot_sem Sem>
   __device__ static inline void fence(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_cluster_t);
