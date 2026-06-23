// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_FENCE_PROXY_ALIAS_H_
#define _CUDA_PTX_GENERATED_FENCE_PROXY_ALIAS_H_

/*
// fence.proxy.alias; // 4. PTX ISA 75, SM_70
template <typename = void>
__device__ static inline void fence_proxy_alias();
*/
#if __cccl_ptx_isa >= 750
template <typename = void>
_CCCL_DEVICE static inline void fence_proxy_alias()
{
  asm volatile("fence.proxy.alias; // 4." : : : "memory");
}
#endif // __cccl_ptx_isa >= 750

/*
// fence.proxy.alias.sem.sys; // PTX ISA 94, SM_90
// .sem       = { .acquire, .release }
template <cuda::ptx::dot_sem Sem>
__device__ static inline void fence_proxy_alias(
  cuda::ptx::sem_t<Sem> sem);
*/
#if __cccl_ptx_isa >= 940
template <::cuda::ptx::dot_sem _Sem>
_CCCL_DEVICE static inline void fence_proxy_alias(::cuda::ptx::sem_t<_Sem> __sem)
{
  static_assert(__sem == sem_acquire || __sem == sem_release, "");
  if constexpr (__sem == sem_acquire)
  {
    asm volatile("fence.proxy.alias.acquire.sys;" : : : "memory");
  }
  else if constexpr (__sem == sem_release)
  {
    asm volatile("fence.proxy.alias.release.sys;" : : : "memory");
  }
}
#endif // __cccl_ptx_isa >= 940

#endif // _CUDA_PTX_GENERATED_FENCE_PROXY_ALIAS_H_
