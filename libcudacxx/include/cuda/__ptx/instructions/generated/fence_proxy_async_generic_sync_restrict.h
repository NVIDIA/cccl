// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_FENCE_PROXY_ASYNC_GENERIC_SYNC_RESTRICT_H_
#define _CUDA_PTX_GENERATED_FENCE_PROXY_ASYNC_GENERIC_SYNC_RESTRICT_H_

/*
// fence.proxy.async::generic.sem.sync_restrict::space.scope; // PTX ISA 86, SM_90
// .sem       = { .acquire }
// .space     = { .shared::cluster }
// .scope     = { .cluster }
template <typename = void>
__device__ static inline void fence_proxy_async_generic_sync_restrict(
  cuda::ptx::sem_acquire_t,
  cuda::ptx::space_cluster_t,
  cuda::ptx::scope_cluster_t);
*/
#if __cccl_ptx_isa >= 860
template <typename = void>
_CCCL_DEVICE static inline void fence_proxy_async_generic_sync_restrict(
  ::cuda::ptx::sem_acquire_t, ::cuda::ptx::space_cluster_t, ::cuda::ptx::scope_cluster_t)
{
  // __sem == sem_acquire (due to parameter type constraint)
  // __space == space_cluster (due to parameter type constraint)
  // __scope == scope_cluster (due to parameter type constraint)
  asm volatile("fence.proxy.async::generic.acquire.sync_restrict::shared::cluster.cluster;" : : : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// fence.proxy.async::generic.sem.sync_restrict::space.scope; // PTX ISA 86, SM_90
// .sem       = { .release }
// .space     = { .shared::cta }
// .scope     = { .cluster }
template <typename = void>
__device__ static inline void fence_proxy_async_generic_sync_restrict(
  cuda::ptx::sem_release_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::scope_cluster_t);
*/
#if __cccl_ptx_isa >= 860
template <typename = void>
_CCCL_DEVICE static inline void fence_proxy_async_generic_sync_restrict(
  ::cuda::ptx::sem_release_t, ::cuda::ptx::space_shared_t, ::cuda::ptx::scope_cluster_t)
{
  // __sem == sem_release (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __scope == scope_cluster (due to parameter type constraint)
  asm volatile("fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster;" : : : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// fence.proxy.async::generic.sem.sync_restrict::space::read.scope; // PTX ISA 94, SM_90
// .sem       = { .release }
// .space     = { .shared::cluster }
// .scope     = { .cta, .cluster }
template <cuda::ptx::dot_scope Scope>
__device__ static inline void fence_proxy_async_generic_sync_restrict(
  cuda::ptx::sem_release_t,
  cuda::ptx::space_cluster_t,
  cuda::ptx::scope_t<Scope> scope);
*/
#if __cccl_ptx_isa >= 940
template <::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline void fence_proxy_async_generic_sync_restrict(
  ::cuda::ptx::sem_release_t, ::cuda::ptx::space_cluster_t, ::cuda::ptx::scope_t<_Scope> __scope)
{
  // __sem == sem_release (due to parameter type constraint)
  // __space == space_cluster (due to parameter type constraint)
  static_assert(__scope == scope_cta || __scope == scope_cluster, "");
  if constexpr (__scope == scope_cta)
  {
    asm volatile("fence.proxy.async::generic.release.sync_restrict::shared::cluster::read.cta;" : : : "memory");
  }
  else if constexpr (__scope == scope_cluster)
  {
    asm volatile("fence.proxy.async::generic.release.sync_restrict::shared::cluster::read.cluster;" : : : "memory");
  }
}
#endif // __cccl_ptx_isa >= 940

#endif // _CUDA_PTX_GENERATED_FENCE_PROXY_ASYNC_GENERIC_SYNC_RESTRICT_H_
