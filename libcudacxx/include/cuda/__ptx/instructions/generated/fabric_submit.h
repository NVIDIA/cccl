// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_FABRIC_SUBMIT_H_
#define _CUDA_PTX_GENERATED_FABRIC_SUBMIT_H_

/*
// fabric.submit; // PTX ISA 93, SM_100
template <typename = void>
__device__ static inline void fabric_submit();
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_submit()
{
  asm volatile("fabric.submit;" : : : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.submit.op_restrict::fetching; // PTX ISA 93, SM_100
template <typename = void>
__device__ static inline void fabric_submit_op_restrict_fetching();
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_submit_op_restrict_fetching()
{
  asm volatile("fabric.submit.op_restrict::fetching;" : : : "memory");
}
#endif // __cccl_ptx_isa >= 930

#endif // _CUDA_PTX_GENERATED_FABRIC_SUBMIT_H_
