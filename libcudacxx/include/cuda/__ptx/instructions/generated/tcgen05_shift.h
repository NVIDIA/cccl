// This file was automatically generated. Do not edit.

// clang-tidy does not distinguish generated PTX constraints or inline-assembly branch bodies.
// NOLINTBEGIN(modernize-unary-static-assert, bugprone-branch-clone)

#ifndef _CUDA_PTX_GENERATED_TCGEN05_SHIFT_H_
#define _CUDA_PTX_GENERATED_TCGEN05_SHIFT_H_

/*
// tcgen05.shift.cta_group.down [taddr]; // PTX ISA 86, SM_100a, SM_103a, SM_107a, SM_110a
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_shift_down(
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE_API void tcgen05_shift_down(::cuda::ptx::cta_group_t<_Cta_Group> __cta_group, ::cuda::std::uint32_t __taddr)
{
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
  if constexpr (__cta_group == cta_group_1)
  {
    asm volatile("tcgen05.shift.cta_group::1.down [%0];" : : "r"(__taddr) : "memory");
  }
  else if constexpr (__cta_group == cta_group_2)
  {
    asm volatile("tcgen05.shift.cta_group::2.down [%0];" : : "r"(__taddr) : "memory");
  }
}
#endif // __cccl_ptx_isa >= 860

// NOLINTEND(modernize-unary-static-assert, bugprone-branch-clone)

#endif // _CUDA_PTX_GENERATED_TCGEN05_SHIFT_H_
