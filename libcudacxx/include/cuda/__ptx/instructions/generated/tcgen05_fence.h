// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_TCGEN05_FENCE_H_
#define _CUDA_PTX_GENERATED_TCGEN05_FENCE_H_

/*
// tcgen05.fence::before_thread_sync; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
template <typename = void>
__device__ static inline void tcgen05_fence_before_thread_sync();
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_fence_before_thread_sync_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename = void>
_CCCL_DEVICE static inline void tcgen05_fence_before_thread_sync()
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm volatile("tcgen05.fence::before_thread_sync;" : : : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_fence_before_thread_sync_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.fence::after_thread_sync; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
template <typename = void>
__device__ static inline void tcgen05_fence_after_thread_sync();
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_fence_after_thread_sync_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename = void>
_CCCL_DEVICE static inline void tcgen05_fence_after_thread_sync()
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm volatile("tcgen05.fence::after_thread_sync;" : : : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_fence_after_thread_sync_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

#endif // _CUDA_PTX_GENERATED_TCGEN05_FENCE_H_
