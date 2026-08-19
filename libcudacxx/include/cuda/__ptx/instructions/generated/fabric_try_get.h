// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_FABRIC_TRY_GET_H_
#define _CUDA_PTX_GENERATED_FABRIC_TRY_GET_H_

/*
// fabric.try_get.async.dst.mbarrier::complete_tx::bytes.mbarrier::report::fabric.sem.scope.b128 [dstMem], [srcLeId,
srcDataOff], size, [smem_bar]; // PTX ISA 93, SM_100
// .dst       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_get(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  void* dstMem,
  uint32_t srcLeId,
  uint64_t srcDataOff,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_get(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  void* __dstMem,
  ::cuda::std::uint32_t __srcLeId,
  ::cuda::std::uint64_t __srcDataOff,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_get.async.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.b128 [%0], "
      "[%1, %2], %3, [%4];"
      :
      : "r"(__as_ptr_smem(__dstMem)), "r"(__srcLeId), "l"(__srcDataOff), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

#endif // _CUDA_PTX_GENERATED_FABRIC_TRY_GET_H_
