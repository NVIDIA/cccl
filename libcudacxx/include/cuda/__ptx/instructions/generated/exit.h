// This file was automatically generated. Do not edit.

// clang-tidy does not distinguish generated PTX constraints or inline-assembly branch bodies.
// NOLINTBEGIN(modernize-unary-static-assert, bugprone-branch-clone)

#ifndef _CUDA_PTX_GENERATED_EXIT_H_
#define _CUDA_PTX_GENERATED_EXIT_H_

/*
// exit; // PTX ISA 10, SM_50
template <typename = void>
__device__ static inline void exit();
*/
#if __cccl_ptx_isa >= 100
template <typename = void>
_CCCL_DEVICE_API void exit()
{
  asm volatile("exit;" : : :);
}
#endif // __cccl_ptx_isa >= 100

// NOLINTEND(modernize-unary-static-assert, bugprone-branch-clone)

#endif // _CUDA_PTX_GENERATED_EXIT_H_
