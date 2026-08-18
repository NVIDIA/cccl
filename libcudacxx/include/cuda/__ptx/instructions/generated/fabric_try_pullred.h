// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_FABRIC_TRY_PULLRED_H_
#define _CUDA_PTX_GENERATED_FABRIC_TRY_PULLRED_H_

/*
// fabric.try_pullred.async.multimem.dst.mbarrier::complete_tx::bytes.mbarrier::report::fabric.sem.scope.and.b32.sync
[dst], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF; // PTX ISA 94, SM_100
// .op        = { .and }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .dst       = { .shared::cta }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void fabric_try_pullred(
  cuda::ptx::op_and_op_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::space_shared_t,
  B32* dst,
  uint32_t srcLeId,
  uint64_t srcDataOff,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void fabric_try_pullred(
  ::cuda::ptx::op_and_op_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::space_shared_t,
  _B32* __dst,
  ::cuda::std::uint32_t __srcLeId,
  ::cuda::std::uint64_t __srcDataOff,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_and_op (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  asm("fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys."
      "and.b32.sync [%0], [%1, %2], %3, [%4], 0xFFFFFFFF;"
      :
      : "r"(__as_ptr_smem(__dst)), "r"(__srcLeId), "l"(__srcDataOff), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_pullred.async.multimem.dst.mbarrier::complete_tx::bytes.mbarrier::report::fabric.sem.scope.xor.b32.sync
[dst], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF; // PTX ISA 94, SM_100
// .op        = { .xor }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .dst       = { .shared::cta }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void fabric_try_pullred(
  cuda::ptx::op_xor_op_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::space_shared_t,
  B32* dst,
  uint32_t srcLeId,
  uint64_t srcDataOff,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void fabric_try_pullred(
  ::cuda::ptx::op_xor_op_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::space_shared_t,
  _B32* __dst,
  ::cuda::std::uint32_t __srcLeId,
  ::cuda::std::uint64_t __srcDataOff,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_xor_op (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  asm("fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys."
      "xor.b32.sync [%0], [%1, %2], %3, [%4], 0xFFFFFFFF;"
      :
      : "r"(__as_ptr_smem(__dst)), "r"(__srcLeId), "l"(__srcDataOff), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_pullred.async.multimem.dst.mbarrier::complete_tx::bytes.mbarrier::report::fabric.sem.scope.or.b32.sync
[dst], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF; // PTX ISA 94, SM_100
// .op        = { .or }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .dst       = { .shared::cta }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void fabric_try_pullred(
  cuda::ptx::op_or_op_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::space_shared_t,
  B32* dst,
  uint32_t srcLeId,
  uint64_t srcDataOff,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void fabric_try_pullred(
  ::cuda::ptx::op_or_op_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::space_shared_t,
  _B32* __dst,
  ::cuda::std::uint32_t __srcLeId,
  ::cuda::std::uint64_t __srcDataOff,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_or_op (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  asm("fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys."
      "or.b32.sync [%0], [%1, %2], %3, [%4], 0xFFFFFFFF;"
      :
      : "r"(__as_ptr_smem(__dst)), "r"(__srcLeId), "l"(__srcDataOff), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_pullred.async.multimem.dst.mbarrier::complete_tx::bytes.mbarrier::report::fabric.sem.scope.and.b64.sync
[dst], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF; // PTX ISA 94, SM_100
// .op        = { .and }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .dst       = { .shared::cta }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline void fabric_try_pullred(
  cuda::ptx::op_and_op_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::space_shared_t,
  B64* dst,
  uint32_t srcLeId,
  uint64_t srcDataOff,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline void fabric_try_pullred(
  ::cuda::ptx::op_and_op_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::space_shared_t,
  _B64* __dst,
  ::cuda::std::uint32_t __srcLeId,
  ::cuda::std::uint64_t __srcDataOff,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_and_op (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  asm("fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys."
      "and.b64.sync [%0], [%1, %2], %3, [%4], 0xFFFFFFFF;"
      :
      : "r"(__as_ptr_smem(__dst)), "r"(__srcLeId), "l"(__srcDataOff), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_pullred.async.multimem.dst.mbarrier::complete_tx::bytes.mbarrier::report::fabric.sem.scope.xor.b64.sync
[dst], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF; // PTX ISA 94, SM_100
// .op        = { .xor }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .dst       = { .shared::cta }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline void fabric_try_pullred(
  cuda::ptx::op_xor_op_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::space_shared_t,
  B64* dst,
  uint32_t srcLeId,
  uint64_t srcDataOff,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline void fabric_try_pullred(
  ::cuda::ptx::op_xor_op_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::space_shared_t,
  _B64* __dst,
  ::cuda::std::uint32_t __srcLeId,
  ::cuda::std::uint64_t __srcDataOff,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_xor_op (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  asm("fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys."
      "xor.b64.sync [%0], [%1, %2], %3, [%4], 0xFFFFFFFF;"
      :
      : "r"(__as_ptr_smem(__dst)), "r"(__srcLeId), "l"(__srcDataOff), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_pullred.async.multimem.dst.mbarrier::complete_tx::bytes.mbarrier::report::fabric.sem.scope.or.b64.sync
[dst], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF; // PTX ISA 94, SM_100
// .op        = { .or }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .dst       = { .shared::cta }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline void fabric_try_pullred(
  cuda::ptx::op_or_op_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::space_shared_t,
  B64* dst,
  uint32_t srcLeId,
  uint64_t srcDataOff,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline void fabric_try_pullred(
  ::cuda::ptx::op_or_op_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::space_shared_t,
  _B64* __dst,
  ::cuda::std::uint32_t __srcLeId,
  ::cuda::std::uint64_t __srcDataOff,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_or_op (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  asm("fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys."
      "or.b64.sync [%0], [%1, %2], %3, [%4], 0xFFFFFFFF;"
      :
      : "r"(__as_ptr_smem(__dst)), "r"(__srcLeId), "l"(__srcDataOff), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_pullred.async.multimem.dst.mbarrier::complete_tx::bytes.mbarrier::report::fabric.sem.scope.min.u32.sync
[dst], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF; // PTX ISA 94, SM_100
// .op        = { .min }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .dst       = { .shared::cta }
template <typename = void>
__device__ static inline void fabric_try_pullred(
  cuda::ptx::op_min_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::space_shared_t,
  uint32_t* dst,
  uint32_t srcLeId,
  uint64_t srcDataOff,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_pullred(
  ::cuda::ptx::op_min_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::std::uint32_t* __dst,
  ::cuda::std::uint32_t __srcLeId,
  ::cuda::std::uint64_t __srcDataOff,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_min (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  asm("fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys."
      "min.u32.sync [%0], [%1, %2], %3, [%4], 0xFFFFFFFF;"
      :
      : "r"(__as_ptr_smem(__dst)), "r"(__srcLeId), "l"(__srcDataOff), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_pullred.async.multimem.dst.mbarrier::complete_tx::bytes.mbarrier::report::fabric.sem.scope.max.u32.sync
[dst], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF; // PTX ISA 94, SM_100
// .op        = { .max }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .dst       = { .shared::cta }
template <typename = void>
__device__ static inline void fabric_try_pullred(
  cuda::ptx::op_max_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::space_shared_t,
  uint32_t* dst,
  uint32_t srcLeId,
  uint64_t srcDataOff,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_pullred(
  ::cuda::ptx::op_max_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::std::uint32_t* __dst,
  ::cuda::std::uint32_t __srcLeId,
  ::cuda::std::uint64_t __srcDataOff,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_max (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  asm("fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys."
      "max.u32.sync [%0], [%1, %2], %3, [%4], 0xFFFFFFFF;"
      :
      : "r"(__as_ptr_smem(__dst)), "r"(__srcLeId), "l"(__srcDataOff), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_pullred.async.multimem.dst.mbarrier::complete_tx::bytes.mbarrier::report::fabric.sem.scope.min.s32.sync
[dst], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF; // PTX ISA 94, SM_100
// .op        = { .min }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .dst       = { .shared::cta }
template <typename = void>
__device__ static inline void fabric_try_pullred(
  cuda::ptx::op_min_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::space_shared_t,
  int32_t* dst,
  uint32_t srcLeId,
  uint64_t srcDataOff,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_pullred(
  ::cuda::ptx::op_min_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::std::int32_t* __dst,
  ::cuda::std::uint32_t __srcLeId,
  ::cuda::std::uint64_t __srcDataOff,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_min (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  asm("fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys."
      "min.s32.sync [%0], [%1, %2], %3, [%4], 0xFFFFFFFF;"
      :
      : "r"(__as_ptr_smem(__dst)), "r"(__srcLeId), "l"(__srcDataOff), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_pullred.async.multimem.dst.mbarrier::complete_tx::bytes.mbarrier::report::fabric.sem.scope.max.s32.sync
[dst], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF; // PTX ISA 94, SM_100
// .op        = { .max }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .dst       = { .shared::cta }
template <typename = void>
__device__ static inline void fabric_try_pullred(
  cuda::ptx::op_max_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::space_shared_t,
  int32_t* dst,
  uint32_t srcLeId,
  uint64_t srcDataOff,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_pullred(
  ::cuda::ptx::op_max_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::std::int32_t* __dst,
  ::cuda::std::uint32_t __srcLeId,
  ::cuda::std::uint64_t __srcDataOff,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_max (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  asm("fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys."
      "max.s32.sync [%0], [%1, %2], %3, [%4], 0xFFFFFFFF;"
      :
      : "r"(__as_ptr_smem(__dst)), "r"(__srcLeId), "l"(__srcDataOff), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_pullred.async.multimem.dst.mbarrier::complete_tx::bytes.mbarrier::report::fabric.sem.scope.min.u64.sync
[dst], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF; // PTX ISA 94, SM_100
// .op        = { .min }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .dst       = { .shared::cta }
template <typename = void>
__device__ static inline void fabric_try_pullred(
  cuda::ptx::op_min_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::space_shared_t,
  uint64_t* dst,
  uint32_t srcLeId,
  uint64_t srcDataOff,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_pullred(
  ::cuda::ptx::op_min_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::std::uint64_t* __dst,
  ::cuda::std::uint32_t __srcLeId,
  ::cuda::std::uint64_t __srcDataOff,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_min (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  asm("fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys."
      "min.u64.sync [%0], [%1, %2], %3, [%4], 0xFFFFFFFF;"
      :
      : "r"(__as_ptr_smem(__dst)), "r"(__srcLeId), "l"(__srcDataOff), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_pullred.async.multimem.dst.mbarrier::complete_tx::bytes.mbarrier::report::fabric.sem.scope.max.u64.sync
[dst], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF; // PTX ISA 94, SM_100
// .op        = { .max }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .dst       = { .shared::cta }
template <typename = void>
__device__ static inline void fabric_try_pullred(
  cuda::ptx::op_max_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::space_shared_t,
  uint64_t* dst,
  uint32_t srcLeId,
  uint64_t srcDataOff,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_pullred(
  ::cuda::ptx::op_max_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::std::uint64_t* __dst,
  ::cuda::std::uint32_t __srcLeId,
  ::cuda::std::uint64_t __srcDataOff,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_max (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  asm("fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys."
      "max.u64.sync [%0], [%1, %2], %3, [%4], 0xFFFFFFFF;"
      :
      : "r"(__as_ptr_smem(__dst)), "r"(__srcLeId), "l"(__srcDataOff), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_pullred.async.multimem.dst.mbarrier::complete_tx::bytes.mbarrier::report::fabric.sem.scope.min.s64.sync
[dst], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF; // PTX ISA 94, SM_100
// .op        = { .min }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .dst       = { .shared::cta }
template <typename = void>
__device__ static inline void fabric_try_pullred(
  cuda::ptx::op_min_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::space_shared_t,
  int64_t* dst,
  uint32_t srcLeId,
  uint64_t srcDataOff,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_pullred(
  ::cuda::ptx::op_min_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::std::int64_t* __dst,
  ::cuda::std::uint32_t __srcLeId,
  ::cuda::std::uint64_t __srcDataOff,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_min (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  asm("fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys."
      "min.s64.sync [%0], [%1, %2], %3, [%4], 0xFFFFFFFF;"
      :
      : "r"(__as_ptr_smem(__dst)), "r"(__srcLeId), "l"(__srcDataOff), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_pullred.async.multimem.dst.mbarrier::complete_tx::bytes.mbarrier::report::fabric.sem.scope.max.s64.sync
[dst], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF; // PTX ISA 94, SM_100
// .op        = { .max }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .dst       = { .shared::cta }
template <typename = void>
__device__ static inline void fabric_try_pullred(
  cuda::ptx::op_max_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::space_shared_t,
  int64_t* dst,
  uint32_t srcLeId,
  uint64_t srcDataOff,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_pullred(
  ::cuda::ptx::op_max_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::std::int64_t* __dst,
  ::cuda::std::uint32_t __srcLeId,
  ::cuda::std::uint64_t __srcDataOff,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_max (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  asm("fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys."
      "max.s64.sync [%0], [%1, %2], %3, [%4], 0xFFFFFFFF;"
      :
      : "r"(__as_ptr_smem(__dst)), "r"(__srcLeId), "l"(__srcDataOff), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_pullred.async.multimem.dst.mbarrier::complete_tx::bytes.mbarrier::report::fabric.sem.scope.min.f16.sync
[dst], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF; // PTX ISA 94, SM_100
// .op        = { .min }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .dst       = { .shared::cta }
template <typename = void>
__device__ static inline void fabric_try_pullred(
  cuda::ptx::op_min_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::space_shared_t,
  __half* dst,
  uint32_t srcLeId,
  uint64_t srcDataOff,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_pullred(
  ::cuda::ptx::op_min_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::space_shared_t,
  __half* __dst,
  ::cuda::std::uint32_t __srcLeId,
  ::cuda::std::uint64_t __srcDataOff,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_min (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  asm("fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys."
      "min.f16.sync [%0], [%1, %2], %3, [%4], 0xFFFFFFFF;"
      :
      : "r"(__as_ptr_smem(__dst)), "r"(__srcLeId), "l"(__srcDataOff), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_pullred.async.multimem.dst.mbarrier::complete_tx::bytes.mbarrier::report::fabric.sem.scope.max.f16.sync
[dst], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF; // PTX ISA 94, SM_100
// .op        = { .max }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .dst       = { .shared::cta }
template <typename = void>
__device__ static inline void fabric_try_pullred(
  cuda::ptx::op_max_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::space_shared_t,
  __half* dst,
  uint32_t srcLeId,
  uint64_t srcDataOff,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_pullred(
  ::cuda::ptx::op_max_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::space_shared_t,
  __half* __dst,
  ::cuda::std::uint32_t __srcLeId,
  ::cuda::std::uint64_t __srcDataOff,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_max (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  asm("fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys."
      "max.f16.sync [%0], [%1, %2], %3, [%4], 0xFFFFFFFF;"
      :
      : "r"(__as_ptr_smem(__dst)), "r"(__srcLeId), "l"(__srcDataOff), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_pullred.async.multimem.dst.mbarrier::complete_tx::bytes.mbarrier::report::fabric.sem.scope.min.bf16.sync
[dst], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF; // PTX ISA 94, SM_100
// .op        = { .min }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .dst       = { .shared::cta }
template <typename = void>
__device__ static inline void fabric_try_pullred(
  cuda::ptx::op_min_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::space_shared_t,
  __nv_bfloat16* dst,
  uint32_t srcLeId,
  uint64_t srcDataOff,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_pullred(
  ::cuda::ptx::op_min_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::space_shared_t,
  __nv_bfloat16* __dst,
  ::cuda::std::uint32_t __srcLeId,
  ::cuda::std::uint64_t __srcDataOff,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_min (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  asm("fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys."
      "min.bf16.sync [%0], [%1, %2], %3, [%4], 0xFFFFFFFF;"
      :
      : "r"(__as_ptr_smem(__dst)), "r"(__srcLeId), "l"(__srcDataOff), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_pullred.async.multimem.dst.mbarrier::complete_tx::bytes.mbarrier::report::fabric.sem.scope.max.bf16.sync
[dst], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF; // PTX ISA 94, SM_100
// .op        = { .max }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .dst       = { .shared::cta }
template <typename = void>
__device__ static inline void fabric_try_pullred(
  cuda::ptx::op_max_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::space_shared_t,
  __nv_bfloat16* dst,
  uint32_t srcLeId,
  uint64_t srcDataOff,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_pullred(
  ::cuda::ptx::op_max_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::space_shared_t,
  __nv_bfloat16* __dst,
  ::cuda::std::uint32_t __srcLeId,
  ::cuda::std::uint64_t __srcDataOff,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_max (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  asm("fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys."
      "max.bf16.sync [%0], [%1, %2], %3, [%4], 0xFFFFFFFF;"
      :
      : "r"(__as_ptr_smem(__dst)), "r"(__srcLeId), "l"(__srcDataOff), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_pullred.async.multimem.dst.mbarrier::complete_tx::bytes.mbarrier::report::fabric.sem.scope.add.u32.sync
[dst], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF; // PTX ISA 94, SM_100
// .op        = { .add }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .dst       = { .shared::cta }
template <typename = void>
__device__ static inline void fabric_try_pullred(
  cuda::ptx::op_add_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::space_shared_t,
  uint32_t* dst,
  uint32_t srcLeId,
  uint64_t srcDataOff,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_pullred(
  ::cuda::ptx::op_add_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::std::uint32_t* __dst,
  ::cuda::std::uint32_t __srcLeId,
  ::cuda::std::uint64_t __srcDataOff,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_add (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  asm("fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys."
      "add.u32.sync [%0], [%1, %2], %3, [%4], 0xFFFFFFFF;"
      :
      : "r"(__as_ptr_smem(__dst)), "r"(__srcLeId), "l"(__srcDataOff), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_pullred.async.multimem.dst.mbarrier::complete_tx::bytes.mbarrier::report::fabric.sem.scope.add.u64.sync
[dst], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF; // PTX ISA 94, SM_100
// .op        = { .add }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .dst       = { .shared::cta }
template <typename = void>
__device__ static inline void fabric_try_pullred(
  cuda::ptx::op_add_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::space_shared_t,
  uint64_t* dst,
  uint32_t srcLeId,
  uint64_t srcDataOff,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_pullred(
  ::cuda::ptx::op_add_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::std::uint64_t* __dst,
  ::cuda::std::uint32_t __srcLeId,
  ::cuda::std::uint64_t __srcDataOff,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_add (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  asm("fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys."
      "add.u64.sync [%0], [%1, %2], %3, [%4], 0xFFFFFFFF;"
      :
      : "r"(__as_ptr_smem(__dst)), "r"(__srcLeId), "l"(__srcDataOff), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_pullred.async.multimem.dst.mbarrier::complete_tx::bytes.mbarrier::report::fabric.sem.scope.add.f16.sync
[dst], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF; // PTX ISA 94, SM_100
// .op        = { .add }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .dst       = { .shared::cta }
template <typename = void>
__device__ static inline void fabric_try_pullred(
  cuda::ptx::op_add_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::space_shared_t,
  __half* dst,
  uint32_t srcLeId,
  uint64_t srcDataOff,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_pullred(
  ::cuda::ptx::op_add_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::space_shared_t,
  __half* __dst,
  ::cuda::std::uint32_t __srcLeId,
  ::cuda::std::uint64_t __srcDataOff,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_add (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  asm("fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys."
      "add.f16.sync [%0], [%1, %2], %3, [%4], 0xFFFFFFFF;"
      :
      : "r"(__as_ptr_smem(__dst)), "r"(__srcLeId), "l"(__srcDataOff), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_pullred.async.multimem.dst.mbarrier::complete_tx::bytes.mbarrier::report::fabric.sem.scope.add.bf16.sync
[dst], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF; // PTX ISA 94, SM_100
// .op        = { .add }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .dst       = { .shared::cta }
template <typename = void>
__device__ static inline void fabric_try_pullred(
  cuda::ptx::op_add_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::space_shared_t,
  __nv_bfloat16* dst,
  uint32_t srcLeId,
  uint64_t srcDataOff,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_pullred(
  ::cuda::ptx::op_add_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::space_shared_t,
  __nv_bfloat16* __dst,
  ::cuda::std::uint32_t __srcLeId,
  ::cuda::std::uint64_t __srcDataOff,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_add (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  asm("fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys."
      "add.bf16.sync [%0], [%1, %2], %3, [%4], 0xFFFFFFFF;"
      :
      : "r"(__as_ptr_smem(__dst)), "r"(__srcLeId), "l"(__srcDataOff), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// fabric.try_pullred.async.multimem.dst.mbarrier::complete_tx::bytes.mbarrier::report::fabric.sem.scope.add.f32.sync
[dst], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF; // PTX ISA 94, SM_100
// .op        = { .add }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .dst       = { .shared::cta }
template <typename = void>
__device__ static inline void fabric_try_pullred(
  cuda::ptx::op_add_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::space_shared_t,
  float* dst,
  uint32_t srcLeId,
  uint64_t srcDataOff,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_pullred(
  ::cuda::ptx::op_add_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::space_shared_t,
  float* __dst,
  ::cuda::std::uint32_t __srcLeId,
  ::cuda::std::uint64_t __srcDataOff,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_add (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  asm("fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys."
      "add.f32.sync [%0], [%1, %2], %3, [%4], 0xFFFFFFFF;"
      :
      : "r"(__as_ptr_smem(__dst)), "r"(__srcLeId), "l"(__srcDataOff), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
//
fabric.try_pullred.async.multimem.dst.mbarrier::complete_tx::bytes.mbarrier::report::fabric.sem.scope.add.acc::f32.f16.sync
[dst_mem], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF; // PTX ISA 93, SM_100a, SM_100f, SM_103a, SM_103f,
SM_107a, SM_107f, SM_110a, SM_110f, SM_120a, SM_120f, SM_121a, SM_121f
// .op        = { .add }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .dst       = { .shared::cta }
template <typename = void>
__device__ static inline void fabric_try_pullred_acc_f32(
  cuda::ptx::op_add_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::space_shared_t,
  __half* dst_mem,
  uint32_t srcLeId,
  uint64_t srcDataOff,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_pullred_acc_f32(
  ::cuda::ptx::op_add_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::space_shared_t,
  __half* __dst_mem,
  ::cuda::std::uint32_t __srcLeId,
  ::cuda::std::uint64_t __srcDataOff,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_add (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  asm("fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys."
      "add.acc::f32.f16.sync [%0], [%1, %2], %3, [%4], 0xFFFFFFFF;"
      :
      : "r"(__as_ptr_smem(__dst_mem)), "r"(__srcLeId), "l"(__srcDataOff), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
//
fabric.try_pullred.async.multimem.dst.mbarrier::complete_tx::bytes.mbarrier::report::fabric.sem.scope.add.acc::f32.bf16.sync
[dst_mem], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF; // PTX ISA 93, SM_100a, SM_100f, SM_103a, SM_103f,
SM_107a, SM_107f, SM_110a, SM_110f, SM_120a, SM_120f, SM_121a, SM_121f
// .op        = { .add }
// .sem       = { .relaxed }
// .scope     = { .sys }
// .dst       = { .shared::cta }
template <typename = void>
__device__ static inline void fabric_try_pullred_acc_f32(
  cuda::ptx::op_add_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  cuda::ptx::space_shared_t,
  __nv_bfloat16* dst_mem,
  uint32_t srcLeId,
  uint64_t srcDataOff,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_pullred_acc_f32(
  ::cuda::ptx::op_add_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::ptx::space_shared_t,
  __nv_bfloat16* __dst_mem,
  ::cuda::std::uint32_t __srcLeId,
  ::cuda::std::uint64_t __srcDataOff,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_add (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  asm("fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys."
      "add.acc::f32.bf16.sync [%0], [%1, %2], %3, [%4], 0xFFFFFFFF;"
      :
      : "r"(__as_ptr_smem(__dst_mem)), "r"(__srcLeId), "l"(__srcDataOff), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

#endif // _CUDA_PTX_GENERATED_FABRIC_TRY_PULLRED_H_
