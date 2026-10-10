..
   This file was automatically generated. Do not edit.

fence.proxy.fabric::generic.alias.acquire.sys
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fence.proxy.fabric::generic.alias.sem.sys; // PTX ISA 93, SM_100
   // .sem       = { .acquire, .release }
   template <cuda::ptx::dot_sem Sem>
   __device__ static inline void fence_proxy_fabric_generic_alias(
     cuda::ptx::sem_t<Sem> sem);

fence.proxy.fabric::generic.alias.release.sys
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fence.proxy.fabric::generic.alias.sem.sys; // PTX ISA 93, SM_100
   // .sem       = { .acquire, .release }
   template <cuda::ptx::dot_sem Sem>
   __device__ static inline void fence_proxy_fabric_generic_alias(
     cuda::ptx::sem_t<Sem> sem);
