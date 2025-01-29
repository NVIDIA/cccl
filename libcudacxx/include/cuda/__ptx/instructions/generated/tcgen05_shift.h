// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_TCGEN05_SHIFT_H_
#define _CUDA_PTX_GENERATED_TCGEN05_SHIFT_H_

/*
// tcgen05.shift.cta_group.down [taddr]; // PTX ISA 86, SM_100a, SM_101a
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_shift_down(
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_shift_down_is_not_supported_before_SM_100a_SM_101a__();
template <dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_shift_down(cta_group_t<_Cta_Group> __cta_group, _CUDA_VSTD::uint32_t __taddr)
{
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__cta_group == cta_group_1)
  {
    asm volatile("tcgen05.shift.cta_group::1.down [%0];" : : "r"(__taddr) : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__cta_group == cta_group_2)
  {
    asm volatile("tcgen05.shift.cta_group::2.down [%0];" : : "r"(__taddr) : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_shift_down_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

#endif // _CUDA_PTX_GENERATED_TCGEN05_SHIFT_H_
