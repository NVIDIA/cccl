..
   This file was automatically generated. Do not edit.

fence.proxy.tensormap::generic.release.cta
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fence.proxy.tensormap::generic.release.scope; // 7. PTX ISA 83, SM_90
   // .sem       = { .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline void fence_proxy_tensormap_generic(
     cuda::ptx::sem_release_t,
     cuda::ptx::scope_t<Scope> scope);

fence.proxy.tensormap::generic.release.cluster
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fence.proxy.tensormap::generic.release.scope; // 7. PTX ISA 83, SM_90
   // .sem       = { .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline void fence_proxy_tensormap_generic(
     cuda::ptx::sem_release_t,
     cuda::ptx::scope_t<Scope> scope);

fence.proxy.tensormap::generic.release.gpu
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fence.proxy.tensormap::generic.release.scope; // 7. PTX ISA 83, SM_90
   // .sem       = { .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline void fence_proxy_tensormap_generic(
     cuda::ptx::sem_release_t,
     cuda::ptx::scope_t<Scope> scope);

fence.proxy.tensormap::generic.release.sys
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fence.proxy.tensormap::generic.release.scope; // 7. PTX ISA 83, SM_90
   // .sem       = { .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   template <cuda::ptx::dot_scope Scope>
   __device__ static inline void fence_proxy_tensormap_generic(
     cuda::ptx::sem_release_t,
     cuda::ptx::scope_t<Scope> scope);

fence.proxy.tensormap::generic.acquire.cta
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

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
.. code-block:: cuda

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
.. code-block:: cuda

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
.. code-block:: cuda

   // fence.proxy.tensormap::generic.sem.scope [addr], size; // 8. PTX ISA 83, SM_90
   // .sem       = { .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   template <int N32, cuda::ptx::dot_scope Scope>
   __device__ static inline void fence_proxy_tensormap_generic(
     cuda::ptx::sem_acquire_t,
     cuda::ptx::scope_t<Scope> scope,
     const void* addr,
     cuda::ptx::n32_t<N32> size);
