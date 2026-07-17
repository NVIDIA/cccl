// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_EXIT_H_
#define _CUDA_PTX_GENERATED_EXIT_H_

/*
// exit; // PTX ISA 10, SM_50
template <typename = void>
__device__ static inline void exit();
*/
#if __cccl_ptx_isa >= 100
template <typename = void>
_CCCL_DEVICE static inline void exit()
{
  asm volatile("exit;" : : :);
}
#endif // __cccl_ptx_isa >= 100

#endif // _CUDA_PTX_GENERATED_EXIT_H_
