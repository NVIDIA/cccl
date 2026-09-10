// This file was automatically generated. Do not edit.

// clang-tidy does not distinguish generated PTX constraints or inline-assembly branch bodies.
// NOLINTBEGIN(modernize-unary-static-assert, bugprone-branch-clone)

#ifndef _CUDA_PTX_GENERATED_BMSK_H_
#define _CUDA_PTX_GENERATED_BMSK_H_

/*
// bmsk.clamp.b32 dest, a_reg, b_reg; // PTX ISA 76, SM_70
template <typename = void>
__device__ static inline uint32_t bmsk_clamp(
  uint32_t a_reg,
  uint32_t b_reg);
*/
#if __cccl_ptx_isa >= 760
template <typename = void>
_CCCL_DEVICE_API ::cuda::std::uint32_t bmsk_clamp(::cuda::std::uint32_t __a_reg, ::cuda::std::uint32_t __b_reg)
{
  ::cuda::std::uint32_t __dest;
  asm("bmsk.clamp.b32 %0, %1, %2;" : "=r"(__dest) : "r"(__a_reg), "r"(__b_reg) :);
  return __dest;
}
#endif // __cccl_ptx_isa >= 760

/*
// bmsk.wrap.b32 dest, a_reg, b_reg; // PTX ISA 76, SM_70
template <typename = void>
__device__ static inline uint32_t bmsk_wrap(
  uint32_t a_reg,
  uint32_t b_reg);
*/
#if __cccl_ptx_isa >= 760
template <typename = void>
_CCCL_DEVICE_API ::cuda::std::uint32_t bmsk_wrap(::cuda::std::uint32_t __a_reg, ::cuda::std::uint32_t __b_reg)
{
  ::cuda::std::uint32_t __dest;
  asm("bmsk.wrap.b32 %0, %1, %2;" : "=r"(__dest) : "r"(__a_reg), "r"(__b_reg) :);
  return __dest;
}
#endif // __cccl_ptx_isa >= 760

// NOLINTEND(modernize-unary-static-assert, bugprone-branch-clone)

#endif // _CUDA_PTX_GENERATED_BMSK_H_
