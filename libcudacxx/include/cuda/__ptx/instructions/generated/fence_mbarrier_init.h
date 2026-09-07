// This file was automatically generated. Do not edit.

// clang-tidy does not distinguish generated PTX constraints or inline-assembly branch bodies.
// NOLINTBEGIN(modernize-unary-static-assert, bugprone-branch-clone)

#ifndef _CUDA_PTX_GENERATED_FENCE_MBARRIER_INIT_H_
#define _CUDA_PTX_GENERATED_FENCE_MBARRIER_INIT_H_

/*
// fence.mbarrier_init.sem.scope; // 3. PTX ISA 80, SM_90
// .sem       = { .release }
// .scope     = { .cluster }
template <typename = void>
__device__ static inline void fence_mbarrier_init(
  cuda::ptx::sem_release_t,
  cuda::ptx::scope_cluster_t);
*/
#if __cccl_ptx_isa >= 800
template <typename = void>
_CCCL_DEVICE_API void fence_mbarrier_init(::cuda::ptx::sem_release_t, ::cuda::ptx::scope_cluster_t)
{
  // __sem == sem_release (due to parameter type constraint)
  // __scope == scope_cluster (due to parameter type constraint)
  asm volatile("fence.mbarrier_init.release.cluster; // 3." : : : "memory");
}
#endif // __cccl_ptx_isa >= 800

// NOLINTEND(modernize-unary-static-assert, bugprone-branch-clone)

#endif // _CUDA_PTX_GENERATED_FENCE_MBARRIER_INIT_H_
