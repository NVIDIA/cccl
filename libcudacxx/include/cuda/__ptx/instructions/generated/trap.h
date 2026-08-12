// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_TRAP_H_
#define _CUDA_PTX_GENERATED_TRAP_H_

/*
// trap; // PTX ISA 10, SM_50
template <typename = void>
__device__ static inline void trap();
*/
#if __cccl_ptx_isa >= 100
template <typename = void>
_CCCL_DEVICE static inline void trap()
{
  asm volatile("trap;" : : :);
}
#endif // __cccl_ptx_isa >= 100

#endif // _CUDA_PTX_GENERATED_TRAP_H_
