// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_FENCE_PROXY_ASYNC_H_
#define _CUDA_PTX_GENERATED_FENCE_PROXY_ASYNC_H_

/*
// fence.proxy.async; // 5. PTX ISA 80, SM_90
template <typename = void>
__device__ static inline void fence_proxy_async();
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_fence_proxy_async_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void fence_proxy_async()
{
  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_90,
    (asm volatile("fence.proxy.async; // 5."
                  :
                  :
                  : "memory");),
    (
      // Unsupported architectures will have a linker error with a semi-decent error message
      __cuda_ptx_fence_proxy_async_is_not_supported_before_SM_90__();));
}
#endif // __cccl_ptx_isa >= 800

/*
// fence.proxy.async{.space}; // 6. PTX ISA 80, SM_90
// .space     = { .global, .shared::cluster, .shared::cta }
template <cuda::ptx::dot_space Space>
__device__ static inline void fence_proxy_async(
  cuda::ptx::space_t<Space> space);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_fence_proxy_async_is_not_supported_before_SM_90__();
template <dot_space _Space>
_CCCL_DEVICE static inline void fence_proxy_async(space_t<_Space> __space)
{
  static_assert(__space == space_global || __space == space_cluster || __space == space_shared, "");
  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_90,
    (
      _CCCL_IF_CONSTEXPR (__space == space_global) {
        asm volatile("fence.proxy.async.global; // 6." : : : "memory");
      } else _CCCL_IF_CONSTEXPR (__space == space_cluster) {
        asm volatile("fence.proxy.async.shared::cluster; // 6." : : : "memory");
      } else _CCCL_IF_CONSTEXPR (__space == space_shared) {
        asm volatile("fence.proxy.async.shared::cta; // 6." : : : "memory");
      }),
    (
      // Unsupported architectures will have a linker error with a semi-decent error message
      __cuda_ptx_fence_proxy_async_is_not_supported_before_SM_90__();));
}
#endif // __cccl_ptx_isa >= 800

#endif // _CUDA_PTX_GENERATED_FENCE_PROXY_ASYNC_H_
