// This file was automatically generated. Do not edit.

// clang-tidy does not distinguish generated PTX constraints or inline-assembly branch bodies.
// NOLINTBEGIN(modernize-unary-static-assert, bugprone-branch-clone)

#ifndef _CUDA_PTX_GENERATED_TRAP_H_
#define _CUDA_PTX_GENERATED_TRAP_H_

/*
// trap; // PTX ISA 10, SM_50
template <typename = void>
__device__ static inline void trap();
*/
#if __cccl_ptx_isa >= 100
template <typename = void>
_CCCL_DEVICE_API void trap()
{
  asm volatile("trap;" : : :);
}
#endif // __cccl_ptx_isa >= 100

// NOLINTEND(modernize-unary-static-assert, bugprone-branch-clone)

#endif // _CUDA_PTX_GENERATED_TRAP_H_
