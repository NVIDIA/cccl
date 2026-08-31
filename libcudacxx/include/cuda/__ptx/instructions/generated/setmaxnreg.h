// This file was automatically generated. Do not edit.

// clang-tidy does not distinguish generated PTX constraints or inline-assembly branch bodies.
// NOLINTBEGIN(modernize-unary-static-assert, bugprone-branch-clone)

#ifndef _CUDA_PTX_GENERATED_SETMAXNREG_H_
#define _CUDA_PTX_GENERATED_SETMAXNREG_H_

/*
// setmaxnreg.inc.sync.aligned.u32 imm_reg_count; // PTX ISA 80, SM_90a, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f, SM_120a, SM_120f, SM_121a, SM_121f template <int N32>
__device__ static inline void setmaxnreg_inc(
  cuda::ptx::n32_t<N32> imm_reg_count);
*/
#if __cccl_ptx_isa >= 800
template <int _N32>
_CCCL_DEVICE_API void setmaxnreg_inc(::cuda::ptx::n32_t<_N32> __imm_reg_count)
{
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" : : "n"(__imm_reg_count.value) :);
}
#endif // __cccl_ptx_isa >= 800

/*
// setmaxnreg.dec.sync.aligned.u32 imm_reg_count; // PTX ISA 80, SM_90a, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f, SM_120a, SM_120f, SM_121a, SM_121f template <int N32>
__device__ static inline void setmaxnreg_dec(
  cuda::ptx::n32_t<N32> imm_reg_count);
*/
#if __cccl_ptx_isa >= 800
template <int _N32>
_CCCL_DEVICE_API void setmaxnreg_dec(::cuda::ptx::n32_t<_N32> __imm_reg_count)
{
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" : : "n"(__imm_reg_count.value) :);
}
#endif // __cccl_ptx_isa >= 800

// NOLINTEND(modernize-unary-static-assert, bugprone-branch-clone)

#endif // _CUDA_PTX_GENERATED_SETMAXNREG_H_
