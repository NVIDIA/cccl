// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_SETMAXNREG_H_
#define _CUDA_PTX_GENERATED_SETMAXNREG_H_

/*
// setmaxnreg.inc.sync.aligned.u32 imm_reg_count; // PTX ISA 80, SM_90a, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f, SM_120a, SM_120f, SM_121a, SM_121f template <int N32>
__device__ static inline void setmaxnreg_inc(
  cuda::ptx::n32_t<N32> imm_reg_count);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void
__cuda_ptx_setmaxnreg_inc_is_only_supported_on_SM_90a_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
template <int _N32>
_CCCL_DEVICE static inline void setmaxnreg_inc(::cuda::ptx::n32_t<_N32> __imm_reg_count)
{
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM90_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 900)))   \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FEAT_SM120_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1200))) \
    || (defined(__CUDA_ARCH_FEAT_SM121_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1210))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1200))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1210))
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" : : "n"(__imm_reg_count.value) :);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_setmaxnreg_inc_is_only_supported_on_SM_90a_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

/*
// setmaxnreg.dec.sync.aligned.u32 imm_reg_count; // PTX ISA 80, SM_90a, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f, SM_120a, SM_120f, SM_121a, SM_121f template <int N32>
__device__ static inline void setmaxnreg_dec(
  cuda::ptx::n32_t<N32> imm_reg_count);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void
__cuda_ptx_setmaxnreg_dec_is_only_supported_on_SM_90a_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
template <int _N32>
_CCCL_DEVICE static inline void setmaxnreg_dec(::cuda::ptx::n32_t<_N32> __imm_reg_count)
{
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM90_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 900)))   \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FEAT_SM120_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1200))) \
    || (defined(__CUDA_ARCH_FEAT_SM121_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1210))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1200))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1210))
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" : : "n"(__imm_reg_count.value) :);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_setmaxnreg_dec_is_only_supported_on_SM_90a_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

#endif // _CUDA_PTX_GENERATED_SETMAXNREG_H_
