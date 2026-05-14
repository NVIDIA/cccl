// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_TENSORMAP_QUERY_H_
#define _CUDA_PTX_GENERATED_TENSORMAP_QUERY_H_

/*
// tensormap.query.is_large.u32 outPred, tensorSize; // PTX ISA 94, SM_90a, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f, SM_120a, SM_120f, SM_121a, SM_121f template <typename = void>
__device__ static inline bool tensormap_query_is_large(
  uint32_t tensorSize);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE static inline bool tensormap_query_is_large(::cuda::std::uint32_t __tensorSize)
{
  ::cuda::std::uint32_t __outPred;
  asm("{\n\t"
      ".reg .pred P_OUT; \n\t"
      "tensormap.query.is_large.u32 P_OUT, %1; \n\t"
      "selp.b32 %0, 1, 0, P_OUT; \n"
      "}"
      : "=r"(__outPred)
      : "r"(__tensorSize)
      :);
  return static_cast<bool>(__outPred);
}
#endif // __cccl_ptx_isa >= 940

#endif // _CUDA_PTX_GENERATED_TENSORMAP_QUERY_H_
