..
   This file was automatically generated. Do not edit.

fence.proxy.generic::fabric.alias.acquire.sys
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fence.proxy.generic::fabric.alias.sem.sys; // PTX ISA 93, SM_100
   // .sem       = { .acquire, .release }
   template <cuda::ptx::dot_sem Sem>
   __device__ static inline void fence_proxy_generic_fabric_alias(
     cuda::ptx::sem_t<Sem> sem);

fence.proxy.generic::fabric.alias.release.sys
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fence.proxy.generic::fabric.alias.sem.sys; // PTX ISA 93, SM_100
   // .sem       = { .acquire, .release }
   template <cuda::ptx::dot_sem Sem>
   __device__ static inline void fence_proxy_generic_fabric_alias(
     cuda::ptx::sem_t<Sem> sem);
