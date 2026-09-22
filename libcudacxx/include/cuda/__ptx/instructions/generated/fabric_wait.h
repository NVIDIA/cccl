// This file was automatically generated. Do not edit.

// clang-tidy does not distinguish generated PTX constraints or inline-assembly branch bodies.
// NOLINTBEGIN(modernize-unary-static-assert, bugprone-branch-clone)

#ifndef _CUDA_PTX_GENERATED_FABRIC_WAIT_H_
#define _CUDA_PTX_GENERATED_FABRIC_WAIT_H_

/*
// fabric.wait.sync_restrict::reads; // PTX ISA 93, SM_100
template <typename = void>
__device__ static inline void fabric_wait();
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE_API void fabric_wait()
{
  asm volatile("fabric.wait.sync_restrict::reads;" : : :);
}
#endif // __cccl_ptx_isa >= 930

// NOLINTEND(modernize-unary-static-assert, bugprone-branch-clone)

#endif // _CUDA_PTX_GENERATED_FABRIC_WAIT_H_
