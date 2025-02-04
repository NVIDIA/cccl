// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_TCGEN05_ALLOC_H_
#define _CUDA_PTX_GENERATED_TCGEN05_ALLOC_H_

/*
// tcgen05.alloc.cta_group.sync.aligned.shared::cta.b32 [dst], nCols; // PTX ISA 86, SM_100a, SM_101a
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_alloc(
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t* dst,
  const uint32_t& nCols);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_alloc_is_not_supported_before_SM_100a_SM_101a__();
template <dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void
tcgen05_alloc(cta_group_t<_Cta_Group> __cta_group, _CUDA_VSTD::uint32_t* __dst, const _CUDA_VSTD::uint32_t& __nCols)
{
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__cta_group == cta_group_1)
  {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :
                 : "r"(__as_ptr_smem(__dst)), "r"(__nCols)
                 : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__cta_group == cta_group_2)
  {
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                 :
                 : "r"(__as_ptr_smem(__dst)), "r"(__nCols)
                 : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_alloc_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.dealloc.cta_group.sync.aligned.b32 taddr, nCols; // PTX ISA 86, SM_100a, SM_101a
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_dealloc(
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t taddr,
  const uint32_t& nCols);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_dealloc_is_not_supported_before_SM_100a_SM_101a__();
template <dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void
tcgen05_dealloc(cta_group_t<_Cta_Group> __cta_group, _CUDA_VSTD::uint32_t __taddr, const _CUDA_VSTD::uint32_t& __nCols)
{
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__cta_group == cta_group_1)
  {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" : : "r"(__taddr), "r"(__nCols) : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__cta_group == cta_group_2)
  {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" : : "r"(__taddr), "r"(__nCols) : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_dealloc_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.relinquish_alloc_permit.cta_group.sync.aligned; // PTX ISA 86, SM_100a, SM_101a
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_relinquish_alloc_permit(
  cuda::ptx::cta_group_t<Cta_Group> cta_group);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_relinquish_alloc_permit_is_not_supported_before_SM_100a_SM_101a__();
template <dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_relinquish_alloc_permit(cta_group_t<_Cta_Group> __cta_group)
{
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__cta_group == cta_group_1)
  {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" : : : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__cta_group == cta_group_2)
  {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" : : : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_relinquish_alloc_permit_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

#endif // _CUDA_PTX_GENERATED_TCGEN05_ALLOC_H_
