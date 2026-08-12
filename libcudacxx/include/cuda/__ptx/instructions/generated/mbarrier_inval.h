// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_MBARRIER_INVAL_H_
#define _CUDA_PTX_GENERATED_MBARRIER_INVAL_H_

/*
// mbarrier.inval.shared.b64 [addr]; // PTX ISA 70, SM_80
template <typename = void>
__device__ static inline void mbarrier_inval(
  uint64_t* addr);
*/
#if __cccl_ptx_isa >= 700
template <typename = void>
_CCCL_DEVICE static inline void mbarrier_inval(::cuda::std::uint64_t* __addr)
{
  asm("mbarrier.inval.shared.b64 [%0];" : : "r"(__as_ptr_smem(__addr)) : "memory");
}
#endif // __cccl_ptx_isa >= 700

#endif // _CUDA_PTX_GENERATED_MBARRIER_INVAL_H_
