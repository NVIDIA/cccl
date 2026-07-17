// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_FABRIC_TRY_PUT_H_
#define _CUDA_PTX_GENERATED_FABRIC_TRY_PUT_H_

/*
// fabric.try_put.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.b128 [dstLeId, dstDataOff],
[srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_put(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_put(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_put.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.b128 [%0, %1], "
      "[%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_put.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.cp_mask.sem.scope.b128 [dstLeId,
dstDataOff], [srcMem], size, [smem_bar], bytemask; // PTX ISA 93, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_put_cp_mask(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar,
  uint16_t bytemask);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_put_cp_mask(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint16_t __bytemask)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_put.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.cp_mask.relaxed.sys.b128 "
      "[%0, %1], [%2], %3, [%4], %5;"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar)),
        "h"(__bytemask)
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_put.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.b128 [dstLeId,
dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_put_counted(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_put_counted(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_put.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys."
      "b128 [%0, %1, %2], [%3], %4, [%5];"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "l"(__dstCounterOff),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_put.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.b128 [dstLeId,
dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_put_multimem(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_put_multimem(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_put.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.b128 "
      "[%0, %1], [%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_put.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.cp_mask.sem.scope.b128
[dstLeId, dstDataOff], [srcMem], size, [smem_bar], bytemask; // PTX ISA 93, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_put_multimem_cp_mask(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar,
  uint16_t bytemask);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_put_multimem_cp_mask(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar,
  ::cuda::std::uint16_t __bytemask)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_put.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.cp_mask.relaxed."
      "sys.b128 [%0, %1], [%2], %3, [%4], %5;"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar)),
        "h"(__bytemask)
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_put.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.b128
[dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_put_multimem_counted(
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const void* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_put_multimem_counted(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const void* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_put.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes."
      "relaxed.sys.b128 [%0, %1, %2], [%3], %4, [%5];"
      :
      : "r"(__dstLeId),
        "l"(__dstDataOff),
        "l"(__dstCounterOff),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

#endif // _CUDA_PTX_GENERATED_FABRIC_TRY_PUT_H_
