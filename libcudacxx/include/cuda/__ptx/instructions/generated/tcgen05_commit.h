// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_TCGEN05_COMMIT_H_
#define _CUDA_PTX_GENERATED_TCGEN05_COMMIT_H_

/*
// tcgen05.commit.cta_group.mbarrier::arrive::one.shared::cluster.b64 [smem_bar]; // PTX ISA 86, SM_100a, SM_101a
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_commit(
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_commit_is_not_supported_before_SM_100a_SM_101a__();
template <dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_commit(cta_group_t<_Cta_Group> __cta_group, _CUDA_VSTD::uint64_t* __smem_bar)
{
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__cta_group == cta_group_1)
  {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                 :
                 : "r"(__as_ptr_dsmem(__smem_bar))
                 : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__cta_group == cta_group_2)
  {
    asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                 :
                 : "r"(__as_ptr_dsmem(__smem_bar))
                 : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_commit_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.commit.cta_group.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [smem_bar], ctaMask; // PTX ISA
86, SM_100a, SM_101a
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_commit_multicast(
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint64_t* smem_bar,
  uint16_t ctaMask);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_commit_multicast_is_not_supported_before_SM_100a_SM_101a__();
template <dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_commit_multicast(
  cta_group_t<_Cta_Group> __cta_group, _CUDA_VSTD::uint64_t* __smem_bar, _CUDA_VSTD::uint16_t __ctaMask)
{
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__cta_group == cta_group_1)
  {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
                 :
                 : "r"(__as_ptr_dsmem(__smem_bar)), "h"(__ctaMask)
                 : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__cta_group == cta_group_2)
  {
    asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
                 :
                 : "r"(__as_ptr_dsmem(__smem_bar)), "h"(__ctaMask)
                 : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_commit_multicast_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

#endif // _CUDA_PTX_GENERATED_TCGEN05_COMMIT_H_
