// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_MAPA_H_
#define _CUDA_PTX_GENERATED_MAPA_H_

/*
// mapa.space.u32  dest, addr, target_cta; // PTX ISA 78, SM_90
// .space     = { .shared::cluster }
template <typename Tp>
__device__ static inline Tp* mapa(
  cuda::ptx::space_cluster_t,
  const Tp* addr,
  uint32_t target_cta);
*/
#if __cccl_ptx_isa >= 780
template <typename _Tp>
_CCCL_DEVICE static inline _Tp* mapa(::cuda::ptx::space_cluster_t, const _Tp* __addr, ::cuda::std::uint32_t __target_cta)
{
  // __space == space_cluster (due to parameter type constraint)
  ::cuda::std::uint32_t __dest;
  asm("mapa.shared::cluster.u32  %0, %1, %2;" : "=r"(__dest) : "r"(__as_ptr_smem(__addr)), "r"(__target_cta) :);
  return __from_ptr_dsmem<_Tp>(__dest);
}
#endif // __cccl_ptx_isa >= 780

#endif // _CUDA_PTX_GENERATED_MAPA_H_
