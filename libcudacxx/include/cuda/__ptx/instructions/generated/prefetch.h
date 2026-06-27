// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_PREFETCH_H_
#define _CUDA_PTX_GENERATED_PREFETCH_H_

/*
// prefetch.global.L2 [addr]; // PTX ISA 20, SM_50
template <typename = void>
__device__ static inline void prefetch_L2(cuda::ptx::space_global_t, const void* addr);
*/
#if __cccl_ptx_isa >= 200
extern "C" _CCCL_DEVICE void __cuda_ptx_prefetch_L2_is_not_supported_before_SM_50__();
template <typename = void>
_CCCL_DEVICE static inline void prefetch_L2(space_global_t, const void* __addr)
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 500
  asm volatile("prefetch.global.L2 [%0];" : : "l"(__as_ptr_gmem(__addr)) : "memory");
#  else
  __cuda_ptx_prefetch_L2_is_not_supported_before_SM_50__();
#  endif
}
#endif // __cccl_ptx_isa >= 200

/*
// prefetch.global.L1 [addr]; // PTX ISA 20, SM_50
template <typename = void>
__device__ static inline void prefetch_L1(cuda::ptx::space_global_t, const void* addr);
*/
#if __cccl_ptx_isa >= 200
extern "C" _CCCL_DEVICE void __cuda_ptx_prefetch_L1_is_not_supported_before_SM_50__();
template <typename = void>
_CCCL_DEVICE static inline void prefetch_L1(space_global_t, const void* __addr)
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 500
  asm volatile("prefetch.global.L1 [%0];" : : "l"(__as_ptr_gmem(__addr)) : "memory");
#  else
  __cuda_ptx_prefetch_L1_is_not_supported_before_SM_50__();
#  endif
}
#endif // __cccl_ptx_isa >= 200

#endif // _CUDA_PTX_GENERATED_PREFETCH_H_
