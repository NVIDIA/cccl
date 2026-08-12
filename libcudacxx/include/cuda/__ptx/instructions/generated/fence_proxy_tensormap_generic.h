// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_FENCE_PROXY_TENSORMAP_GENERIC_H_
#define _CUDA_PTX_GENERATED_FENCE_PROXY_TENSORMAP_GENERIC_H_

/*
// fence.proxy.tensormap::generic.release.scope; // 7. PTX ISA 83, SM_90
// .sem       = { .release }
// .scope     = { .cta, .cluster, .gpu, .sys }
template <cuda::ptx::dot_scope Scope>
__device__ static inline void fence_proxy_tensormap_generic(
  cuda::ptx::sem_release_t,
  cuda::ptx::scope_t<Scope> scope);
*/
#if __cccl_ptx_isa >= 830
template <::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline void
fence_proxy_tensormap_generic(::cuda::ptx::sem_release_t, ::cuda::ptx::scope_t<_Scope> __scope)
{
  // __sem == sem_release (due to parameter type constraint)
  static_assert(__scope == scope_cta || __scope == scope_cluster || __scope == scope_gpu || __scope == scope_sys, "");
  if constexpr (__scope == scope_cta)
  {
    asm volatile("fence.proxy.tensormap::generic.release.cta; // 7." : : : "memory");
  }
  else if constexpr (__scope == scope_cluster)
  {
    asm volatile("fence.proxy.tensormap::generic.release.cluster; // 7." : : : "memory");
  }
  else if constexpr (__scope == scope_gpu)
  {
    asm volatile("fence.proxy.tensormap::generic.release.gpu; // 7." : : : "memory");
  }
  else if constexpr (__scope == scope_sys)
  {
    asm volatile("fence.proxy.tensormap::generic.release.sys; // 7." : : : "memory");
  }
}
#endif // __cccl_ptx_isa >= 830

/*
// fence.proxy.tensormap::generic.sem.scope [addr], size; // 8. PTX ISA 83, SM_90
// .sem       = { .acquire }
// .scope     = { .cta, .cluster, .gpu, .sys }
template <int N32, cuda::ptx::dot_scope Scope>
__device__ static inline void fence_proxy_tensormap_generic(
  cuda::ptx::sem_acquire_t,
  cuda::ptx::scope_t<Scope> scope,
  const void* addr,
  cuda::ptx::n32_t<N32> size);
*/
#if __cccl_ptx_isa >= 830
template <int _N32, ::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline void fence_proxy_tensormap_generic(
  ::cuda::ptx::sem_acquire_t, ::cuda::ptx::scope_t<_Scope> __scope, const void* __addr, ::cuda::ptx::n32_t<_N32> __size)
{
  // __sem == sem_acquire (due to parameter type constraint)
  static_assert(__scope == scope_cta || __scope == scope_cluster || __scope == scope_gpu || __scope == scope_sys, "");
  if constexpr (__scope == scope_cta)
  {
    asm volatile("fence.proxy.tensormap::generic.acquire.cta [%0], %1; // 8."
                 :
                 : "l"(__addr), "n"(__size.value)
                 : "memory");
  }
  else if constexpr (__scope == scope_cluster)
  {
    asm volatile("fence.proxy.tensormap::generic.acquire.cluster [%0], %1; // 8."
                 :
                 : "l"(__addr), "n"(__size.value)
                 : "memory");
  }
  else if constexpr (__scope == scope_gpu)
  {
    asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], %1; // 8."
                 :
                 : "l"(__addr), "n"(__size.value)
                 : "memory");
  }
  else if constexpr (__scope == scope_sys)
  {
    asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], %1; // 8."
                 :
                 : "l"(__addr), "n"(__size.value)
                 : "memory");
  }
}
#endif // __cccl_ptx_isa >= 830

#endif // _CUDA_PTX_GENERATED_FENCE_PROXY_TENSORMAP_GENERIC_H_
