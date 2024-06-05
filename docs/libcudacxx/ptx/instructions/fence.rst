.. _libcudacxx-ptx-instructions-fence:

fence
=====

-  PTX ISA:
   `fence <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar-fence>`__

.. _fence-1:

fence
-----

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

fence.mbarrier_init
-------------------

fence.mbarrier_init.release.cluster
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // fence.mbarrier_init.sem.scope; // 3. PTX ISA 80, SM_90
   // .sem       = { .release }
   // .scope     = { .cluster }
   template <typename=void>
   __device__ static inline void fence_mbarrier_init(
     cuda::ptx::sem_release_t,
     cuda::ptx::scope_cluster_t);

fence.proxy.alias
-----------------

fence.proxy.alias
^^^^^^^^^^^^^^^^^
.. code:: cuda

   // fence.proxy.alias; // 4. PTX ISA 75, SM_70
   template <typename=void>
   __device__ static inline void fence_proxy_alias();

fence.proxy.async
-----------------

fence.proxy.async
^^^^^^^^^^^^^^^^^
.. code:: cuda

   // fence.proxy.async; // 5. PTX ISA 80, SM_90
   template <typename=void>
   __device__ static inline void fence_proxy_async();

fence.proxy.async.global
^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // fence.proxy.async{.space}; // 6. PTX ISA 80, SM_90
   // .space     = { .global, .shared::cluster, .shared::cta }
   template <cuda::ptx::dot_space Space>
   __device__ static inline void fence_proxy_async(
     cuda::ptx::space_t<Space> space);

fence.proxy.async.shared::cluster
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // fence.proxy.async{.space}; // 6. PTX ISA 80, SM_90
   // .space     = { .global, .shared::cluster, .shared::cta }
   template <cuda::ptx::dot_space Space>
   __device__ static inline void fence_proxy_async(
     cuda::ptx::space_t<Space> space);

fence.proxy.async.shared::cta
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // fence.proxy.async{.space}; // 6. PTX ISA 80, SM_90
   // .space     = { .global, .shared::cluster, .shared::cta }
   template <cuda::ptx::dot_space Space>
   __device__ static inline void fence_proxy_async(
     cuda::ptx::space_t<Space> space);

fence.proxy.tensormap
---------------------

fence.proxy.tensormap::generic.release.cta
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // fence.proxy.tensormap::generic.release.scope; // 7. PTX ISA 83, SM_90
   // .sem       = { .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline void fence_proxy_tensormap_generic(
     cuda::ptx::sem_release_t,
     cuda::ptx::scope_t<Scope> scope);

fence.proxy.tensormap::generic.release.cluster
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // fence.proxy.tensormap::generic.release.scope; // 7. PTX ISA 83, SM_90
   // .sem       = { .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline void fence_proxy_tensormap_generic(
     cuda::ptx::sem_release_t,
     cuda::ptx::scope_t<Scope> scope);

fence.proxy.tensormap::generic.release.gpu
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // fence.proxy.tensormap::generic.release.scope; // 7. PTX ISA 83, SM_90
   // .sem       = { .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline void fence_proxy_tensormap_generic(
     cuda::ptx::sem_release_t,
     cuda::ptx::scope_t<Scope> scope);

fence.proxy.tensormap::generic.release.sys
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // fence.proxy.tensormap::generic.release.scope; // 7. PTX ISA 83, SM_90
   // .sem       = { .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline void fence_proxy_tensormap_generic(
     cuda::ptx::sem_release_t,
     cuda::ptx::scope_t<Scope> scope);

fence.proxy.tensormap::generic.acquire.cta
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // fence.proxy.tensormap::generic.sem.scope [addr], size; // 8. PTX ISA 83, SM_90
   // .sem       = { .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   template <int N32, cuda::ptx::dot_scope Scope>
   __device__ static inline void fence_proxy_tensormap_generic(
     cuda::ptx::sem_acquire_t,
     cuda::ptx::scope_t<Scope> scope,
     const void* addr,
     cuda::ptx::n32_t<N32> size);

fence.proxy.tensormap::generic.acquire.cluster
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // fence.proxy.tensormap::generic.sem.scope [addr], size; // 8. PTX ISA 83, SM_90
   // .sem       = { .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   template <int N32, cuda::ptx::dot_scope Scope>
   __device__ static inline void fence_proxy_tensormap_generic(
     cuda::ptx::sem_acquire_t,
     cuda::ptx::scope_t<Scope> scope,
     const void* addr,
     cuda::ptx::n32_t<N32> size);

fence.proxy.tensormap::generic.acquire.gpu
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // fence.proxy.tensormap::generic.sem.scope [addr], size; // 8. PTX ISA 83, SM_90
   // .sem       = { .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   template <int N32, cuda::ptx::dot_scope Scope>
   __device__ static inline void fence_proxy_tensormap_generic(
     cuda::ptx::sem_acquire_t,
     cuda::ptx::scope_t<Scope> scope,
     const void* addr,
     cuda::ptx::n32_t<N32> size);

fence.proxy.tensormap::generic.acquire.sys
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // fence.proxy.tensormap::generic.sem.scope [addr], size; // 8. PTX ISA 83, SM_90
   // .sem       = { .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   template <int N32, cuda::ptx::dot_scope Scope>
   __device__ static inline void fence_proxy_tensormap_generic(
     cuda::ptx::sem_acquire_t,
     cuda::ptx::scope_t<Scope> scope,
     const void* addr,
     cuda::ptx::n32_t<N32> size);
