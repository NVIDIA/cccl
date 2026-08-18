// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_FENCE_PROXY_FABRIC_GENERIC_ALIAS_H_
#define _CUDA_PTX_GENERATED_FENCE_PROXY_FABRIC_GENERIC_ALIAS_H_

/*
// fence.proxy.fabric::generic.alias.sem.sys; // PTX ISA 93, SM_100
// .sem       = { .acquire, .release }
template <cuda::ptx::dot_sem Sem>
__device__ static inline void fence_proxy_fabric_generic_alias(
  cuda::ptx::sem_t<Sem> sem);
*/
#if __cccl_ptx_isa >= 930
template <::cuda::ptx::dot_sem _Sem>
_CCCL_DEVICE static inline void fence_proxy_fabric_generic_alias(::cuda::ptx::sem_t<_Sem> __sem)
{
  static_assert(__sem == sem_acquire || __sem == sem_release, "");
  if constexpr (__sem == sem_acquire)
  {
    asm volatile("fence.proxy.fabric::generic.alias.acquire.sys;" : : : "memory");
  }
  else if constexpr (__sem == sem_release)
  {
    asm volatile("fence.proxy.fabric::generic.alias.release.sys;" : : : "memory");
  }
}
#endif // __cccl_ptx_isa >= 930

#endif // _CUDA_PTX_GENERATED_FENCE_PROXY_FABRIC_GENERIC_ALIAS_H_
