// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_TCGEN05_WAIT_H_
#define _CUDA_PTX_GENERATED_TCGEN05_WAIT_H_

/*
// tcgen05.wait::ld.sync.aligned; // PTX ISA 86, SM_100a, SM_101a
template <typename = void>
__device__ static inline void tcgen05_wait_ld();
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_wait_ld_is_not_supported_before_SM_100a_SM_101a__();
template <typename = void>
_CCCL_DEVICE static inline void tcgen05_wait_ld()
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm volatile("tcgen05.wait::ld.sync.aligned;" : : : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_wait_ld_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.wait::st.sync.aligned; // PTX ISA 86, SM_100a, SM_101a
template <typename = void>
__device__ static inline void tcgen05_wait_st();
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_wait_st_is_not_supported_before_SM_100a_SM_101a__();
template <typename = void>
_CCCL_DEVICE static inline void tcgen05_wait_st()
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm volatile("tcgen05.wait::st.sync.aligned;" : : : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_wait_st_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

#endif // _CUDA_PTX_GENERATED_TCGEN05_WAIT_H_
