// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_CP_ASYNC_BULK_MULTICAST_H_
#define _CUDA_PTX_GENERATED_CP_ASYNC_BULK_MULTICAST_H_

/*
// cp.async.bulk.dst.src.mbarrier::complete_tx::bytes.multicast::cluster [dstMem], [srcMem], size, [smem_bar], ctaMask;
// PTX ISA 80, SM_90a, SM_100a, SM_101a
// .dst       = { .shared::cluster }
// .src       = { .global }
template <typename = void>
__device__ static inline void cp_async_bulk(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* srcMem,
  const uint32_t& size,
  uint64_t* smem_bar,
  const uint16_t& ctaMask);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_is_not_supported_before_SM_90a_SM_100a_SM_101a__();
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk(
  space_cluster_t,
  space_global_t,
  void* __dstMem,
  const void* __srcMem,
  const _CUDA_VSTD::uint32_t& __size,
  _CUDA_VSTD::uint64_t* __smem_bar,
  const _CUDA_VSTD::uint16_t& __ctaMask)
{
// __space == space_cluster (due to parameter type constraint)
// __space == space_global (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM90_ALL || __CUDA_ARCH_FEAT_SM100_ALL \
    || __CUDA_ARCH_FEAT_SM101_ALL
  asm("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1], %2, [%3], %4;"
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__as_ptr_gmem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar)),
        "h"(__ctaMask)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_cp_async_bulk_is_not_supported_before_SM_90a_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

#endif // _CUDA_PTX_GENERATED_CP_ASYNC_BULK_MULTICAST_H_
