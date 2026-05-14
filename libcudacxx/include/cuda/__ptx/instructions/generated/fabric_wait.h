// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_FABRIC_WAIT_H_
#define _CUDA_PTX_GENERATED_FABRIC_WAIT_H_

/*
// fabric.wait.sync_restrict::reads; // PTX ISA 93, SM_100
template <typename = void>
__device__ static inline void fabric_wait();
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_wait()
{
  asm volatile("fabric.wait.sync_restrict::reads;" : : :);
}
#endif // __cccl_ptx_isa >= 930

#endif // _CUDA_PTX_GENERATED_FABRIC_WAIT_H_
