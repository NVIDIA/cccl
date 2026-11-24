..
   This file was automatically generated. Do not edit.

fence.proxy.alias
^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fence.proxy.alias; // 4. PTX ISA 75, SM_70
   template <typename = void>
   __device__ static inline void fence_proxy_alias();

fence.proxy.alias.acquire.sys
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fence.proxy.alias.sem.sys; // PTX ISA 94, SM_90
   // .sem       = { .acquire, .release }
   template <cuda::ptx::dot_sem Sem>
   __device__ static inline void fence_proxy_alias(
     cuda::ptx::sem_t<Sem> sem);

fence.proxy.alias.release.sys
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fence.proxy.alias.sem.sys; // PTX ISA 94, SM_90
   // .sem       = { .acquire, .release }
   template <cuda::ptx::dot_sem Sem>
   __device__ static inline void fence_proxy_alias(
     cuda::ptx::sem_t<Sem> sem);
