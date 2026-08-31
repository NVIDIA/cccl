// This file was automatically generated. Do not edit.

// clang-tidy does not distinguish generated PTX constraints or inline-assembly branch bodies.
// NOLINTBEGIN(modernize-unary-static-assert, bugprone-branch-clone)

#ifndef _CUDA_PTX_GENERATED_TCGEN05_FENCE_H_
#define _CUDA_PTX_GENERATED_TCGEN05_FENCE_H_

/*
// tcgen05.fence::before_thread_sync; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a,
SM_110f template <typename = void>
__device__ static inline void tcgen05_fence_before_thread_sync();
*/
#if __cccl_ptx_isa >= 860
template <typename = void>
_CCCL_DEVICE_API void tcgen05_fence_before_thread_sync()
{
  asm volatile("tcgen05.fence::before_thread_sync;" : : : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.fence::after_thread_sync; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a,
SM_110f template <typename = void>
__device__ static inline void tcgen05_fence_after_thread_sync();
*/
#if __cccl_ptx_isa >= 860
template <typename = void>
_CCCL_DEVICE_API void tcgen05_fence_after_thread_sync()
{
  asm volatile("tcgen05.fence::after_thread_sync;" : : : "memory");
}
#endif // __cccl_ptx_isa >= 860

// NOLINTEND(modernize-unary-static-assert, bugprone-branch-clone)

#endif // _CUDA_PTX_GENERATED_TCGEN05_FENCE_H_
