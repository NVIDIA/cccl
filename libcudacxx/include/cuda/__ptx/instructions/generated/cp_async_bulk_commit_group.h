// This file was automatically generated. Do not edit.

// clang-tidy does not distinguish generated PTX constraints or inline-assembly branch bodies.
// NOLINTBEGIN(modernize-unary-static-assert, bugprone-branch-clone)

#ifndef _CUDA_PTX_GENERATED_CP_ASYNC_BULK_COMMIT_GROUP_H_
#define _CUDA_PTX_GENERATED_CP_ASYNC_BULK_COMMIT_GROUP_H_

/*
// cp.async.bulk.commit_group; // PTX ISA 80, SM_90
template <typename = void>
__device__ static inline void cp_async_bulk_commit_group();
*/
#if __cccl_ptx_isa >= 800
template <typename = void>
_CCCL_DEVICE_API void cp_async_bulk_commit_group()
{
  asm volatile("cp.async.bulk.commit_group;" : : :);
}
#endif // __cccl_ptx_isa >= 800

// NOLINTEND(modernize-unary-static-assert, bugprone-branch-clone)

#endif // _CUDA_PTX_GENERATED_CP_ASYNC_BULK_COMMIT_GROUP_H_
