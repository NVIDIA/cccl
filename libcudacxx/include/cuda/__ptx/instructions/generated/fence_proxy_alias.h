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

#endif // _CUDA_PTX_GENERATED_FENCE_PROXY_ALIAS_H_
