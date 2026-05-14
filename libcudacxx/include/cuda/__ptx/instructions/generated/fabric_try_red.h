// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_FABRIC_TRY_RED_H_
#define _CUDA_PTX_GENERATED_FABRIC_TRY_RED_H_

/*
// fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.b32 [dstLeId, dstDataOff],
[srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .and }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void fabric_try_red(
  cuda::ptx::op_and_op_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const B32* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void fabric_try_red(
  ::cuda::ptx::op_and_op_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const _B32* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_and_op (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.and.b32 [%0, "
      "%1], [%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.b32
[dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .and }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void fabric_try_red_counted(
  cuda::ptx::op_and_op_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const B32* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void fabric_try_red_counted(
  ::cuda::ptx::op_and_op_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const _B32* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_and_op (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys."
      "and.b32 [%0, %1, %2], [%3], %4, [%5];"
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
// fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.b32 [dstLeId,
dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .and }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void fabric_try_red_multimem(
  cuda::ptx::op_and_op_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const B32* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void fabric_try_red_multimem(
  ::cuda::ptx::op_and_op_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const _B32* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_and_op (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.and."
      "b32 [%0, %1], [%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.b32
[dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .and }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void fabric_try_red_multimem_counted(
  cuda::ptx::op_and_op_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const B32* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void fabric_try_red_multimem_counted(
  ::cuda::ptx::op_and_op_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const _B32* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_and_op (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes."
      "relaxed.sys.and.b32 [%0, %1, %2], [%3], %4, [%5];"
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
// fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.b32 [dstLeId, dstDataOff],
[srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .xor }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void fabric_try_red(
  cuda::ptx::op_xor_op_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const B32* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void fabric_try_red(
  ::cuda::ptx::op_xor_op_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const _B32* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_xor_op (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.xor.b32 [%0, "
      "%1], [%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.b32
[dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .xor }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void fabric_try_red_counted(
  cuda::ptx::op_xor_op_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const B32* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void fabric_try_red_counted(
  ::cuda::ptx::op_xor_op_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const _B32* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_xor_op (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys."
      "xor.b32 [%0, %1, %2], [%3], %4, [%5];"
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
// fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.b32 [dstLeId,
dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .xor }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void fabric_try_red_multimem(
  cuda::ptx::op_xor_op_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const B32* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void fabric_try_red_multimem(
  ::cuda::ptx::op_xor_op_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const _B32* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_xor_op (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.xor."
      "b32 [%0, %1], [%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.b32
[dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .xor }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void fabric_try_red_multimem_counted(
  cuda::ptx::op_xor_op_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const B32* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void fabric_try_red_multimem_counted(
  ::cuda::ptx::op_xor_op_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const _B32* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_xor_op (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes."
      "relaxed.sys.xor.b32 [%0, %1, %2], [%3], %4, [%5];"
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
// fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.b32 [dstLeId, dstDataOff],
[srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .or }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void fabric_try_red(
  cuda::ptx::op_or_op_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const B32* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void fabric_try_red(
  ::cuda::ptx::op_or_op_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const _B32* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_or_op (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.or.b32 [%0, "
      "%1], [%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.b32
[dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .or }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void fabric_try_red_counted(
  cuda::ptx::op_or_op_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const B32* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void fabric_try_red_counted(
  ::cuda::ptx::op_or_op_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const _B32* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_or_op (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys."
      "or.b32 [%0, %1, %2], [%3], %4, [%5];"
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
// fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.b32 [dstLeId,
dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .or }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void fabric_try_red_multimem(
  cuda::ptx::op_or_op_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const B32* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void fabric_try_red_multimem(
  ::cuda::ptx::op_or_op_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const _B32* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_or_op (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.or."
      "b32 [%0, %1], [%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.b32
[dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .or }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void fabric_try_red_multimem_counted(
  cuda::ptx::op_or_op_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const B32* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void fabric_try_red_multimem_counted(
  ::cuda::ptx::op_or_op_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const _B32* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_or_op (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes."
      "relaxed.sys.or.b32 [%0, %1, %2], [%3], %4, [%5];"
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
// fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.b64 [dstLeId, dstDataOff],
[srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .and }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline void fabric_try_red(
  cuda::ptx::op_and_op_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const B64* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline void fabric_try_red(
  ::cuda::ptx::op_and_op_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const _B64* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_and_op (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.and.b64 [%0, "
      "%1], [%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.b64
[dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .and }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline void fabric_try_red_counted(
  cuda::ptx::op_and_op_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const B64* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline void fabric_try_red_counted(
  ::cuda::ptx::op_and_op_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const _B64* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_and_op (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys."
      "and.b64 [%0, %1, %2], [%3], %4, [%5];"
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
// fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.b64 [dstLeId,
dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .and }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline void fabric_try_red_multimem(
  cuda::ptx::op_and_op_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const B64* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline void fabric_try_red_multimem(
  ::cuda::ptx::op_and_op_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const _B64* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_and_op (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.and."
      "b64 [%0, %1], [%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.b64
[dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .and }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline void fabric_try_red_multimem_counted(
  cuda::ptx::op_and_op_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const B64* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline void fabric_try_red_multimem_counted(
  ::cuda::ptx::op_and_op_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const _B64* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_and_op (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes."
      "relaxed.sys.and.b64 [%0, %1, %2], [%3], %4, [%5];"
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
// fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.b64 [dstLeId, dstDataOff],
[srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .xor }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline void fabric_try_red(
  cuda::ptx::op_xor_op_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const B64* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline void fabric_try_red(
  ::cuda::ptx::op_xor_op_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const _B64* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_xor_op (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.xor.b64 [%0, "
      "%1], [%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.b64
[dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .xor }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline void fabric_try_red_counted(
  cuda::ptx::op_xor_op_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const B64* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline void fabric_try_red_counted(
  ::cuda::ptx::op_xor_op_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const _B64* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_xor_op (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys."
      "xor.b64 [%0, %1, %2], [%3], %4, [%5];"
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
// fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.b64 [dstLeId,
dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .xor }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline void fabric_try_red_multimem(
  cuda::ptx::op_xor_op_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const B64* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline void fabric_try_red_multimem(
  ::cuda::ptx::op_xor_op_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const _B64* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_xor_op (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.xor."
      "b64 [%0, %1], [%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.b64
[dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .xor }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline void fabric_try_red_multimem_counted(
  cuda::ptx::op_xor_op_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const B64* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline void fabric_try_red_multimem_counted(
  ::cuda::ptx::op_xor_op_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const _B64* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_xor_op (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes."
      "relaxed.sys.xor.b64 [%0, %1, %2], [%3], %4, [%5];"
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
// fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.b64 [dstLeId, dstDataOff],
[srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .or }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline void fabric_try_red(
  cuda::ptx::op_or_op_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const B64* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline void fabric_try_red(
  ::cuda::ptx::op_or_op_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const _B64* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_or_op (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.or.b64 [%0, "
      "%1], [%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.b64
[dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .or }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline void fabric_try_red_counted(
  cuda::ptx::op_or_op_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const B64* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline void fabric_try_red_counted(
  ::cuda::ptx::op_or_op_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const _B64* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_or_op (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys."
      "or.b64 [%0, %1, %2], [%3], %4, [%5];"
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
// fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.b64 [dstLeId,
dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .or }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline void fabric_try_red_multimem(
  cuda::ptx::op_or_op_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const B64* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline void fabric_try_red_multimem(
  ::cuda::ptx::op_or_op_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const _B64* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_or_op (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.or."
      "b64 [%0, %1], [%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.b64
[dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .or }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline void fabric_try_red_multimem_counted(
  cuda::ptx::op_or_op_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const B64* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline void fabric_try_red_multimem_counted(
  ::cuda::ptx::op_or_op_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const _B64* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_or_op (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes."
      "relaxed.sys.or.b64 [%0, %1, %2], [%3], %4, [%5];"
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
// fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.u32 [dstLeId, dstDataOff],
[srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .min }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red(
  cuda::ptx::op_min_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const uint32_t* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red(
  ::cuda::ptx::op_min_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const ::cuda::std::uint32_t* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_min (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.min.u32 [%0, "
      "%1], [%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.u32
[dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .min }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_counted(
  cuda::ptx::op_min_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const uint32_t* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_counted(
  ::cuda::ptx::op_min_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const ::cuda::std::uint32_t* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_min (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys."
      "min.u32 [%0, %1, %2], [%3], %4, [%5];"
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
// fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.u32 [dstLeId,
dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .min }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_multimem(
  cuda::ptx::op_min_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const uint32_t* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_multimem(
  ::cuda::ptx::op_min_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const ::cuda::std::uint32_t* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_min (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.min."
      "u32 [%0, %1], [%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.u32
[dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .min }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_multimem_counted(
  cuda::ptx::op_min_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const uint32_t* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_multimem_counted(
  ::cuda::ptx::op_min_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const ::cuda::std::uint32_t* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_min (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes."
      "relaxed.sys.min.u32 [%0, %1, %2], [%3], %4, [%5];"
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
// fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.u32 [dstLeId, dstDataOff],
[srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .max }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red(
  cuda::ptx::op_max_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const uint32_t* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red(
  ::cuda::ptx::op_max_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const ::cuda::std::uint32_t* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_max (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.max.u32 [%0, "
      "%1], [%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.u32
[dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .max }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_counted(
  cuda::ptx::op_max_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const uint32_t* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_counted(
  ::cuda::ptx::op_max_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const ::cuda::std::uint32_t* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_max (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys."
      "max.u32 [%0, %1, %2], [%3], %4, [%5];"
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
// fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.u32 [dstLeId,
dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .max }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_multimem(
  cuda::ptx::op_max_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const uint32_t* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_multimem(
  ::cuda::ptx::op_max_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const ::cuda::std::uint32_t* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_max (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.max."
      "u32 [%0, %1], [%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.u32
[dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .max }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_multimem_counted(
  cuda::ptx::op_max_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const uint32_t* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_multimem_counted(
  ::cuda::ptx::op_max_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const ::cuda::std::uint32_t* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_max (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes."
      "relaxed.sys.max.u32 [%0, %1, %2], [%3], %4, [%5];"
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
// fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.s32 [dstLeId, dstDataOff],
[srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .min }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red(
  cuda::ptx::op_min_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const int32_t* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red(
  ::cuda::ptx::op_min_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const ::cuda::std::int32_t* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_min (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.min.s32 [%0, "
      "%1], [%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.s32
[dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .min }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_counted(
  cuda::ptx::op_min_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const int32_t* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_counted(
  ::cuda::ptx::op_min_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const ::cuda::std::int32_t* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_min (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys."
      "min.s32 [%0, %1, %2], [%3], %4, [%5];"
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
// fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.s32 [dstLeId,
dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .min }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_multimem(
  cuda::ptx::op_min_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const int32_t* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_multimem(
  ::cuda::ptx::op_min_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const ::cuda::std::int32_t* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_min (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.min."
      "s32 [%0, %1], [%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.s32
[dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .min }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_multimem_counted(
  cuda::ptx::op_min_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const int32_t* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_multimem_counted(
  ::cuda::ptx::op_min_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const ::cuda::std::int32_t* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_min (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes."
      "relaxed.sys.min.s32 [%0, %1, %2], [%3], %4, [%5];"
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
// fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.s32 [dstLeId, dstDataOff],
[srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .max }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red(
  cuda::ptx::op_max_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const int32_t* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red(
  ::cuda::ptx::op_max_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const ::cuda::std::int32_t* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_max (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.max.s32 [%0, "
      "%1], [%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.s32
[dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .max }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_counted(
  cuda::ptx::op_max_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const int32_t* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_counted(
  ::cuda::ptx::op_max_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const ::cuda::std::int32_t* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_max (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys."
      "max.s32 [%0, %1, %2], [%3], %4, [%5];"
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
// fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.s32 [dstLeId,
dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .max }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_multimem(
  cuda::ptx::op_max_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const int32_t* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_multimem(
  ::cuda::ptx::op_max_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const ::cuda::std::int32_t* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_max (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.max."
      "s32 [%0, %1], [%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.s32
[dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .max }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_multimem_counted(
  cuda::ptx::op_max_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const int32_t* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_multimem_counted(
  ::cuda::ptx::op_max_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const ::cuda::std::int32_t* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_max (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes."
      "relaxed.sys.max.s32 [%0, %1, %2], [%3], %4, [%5];"
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
// fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.u64 [dstLeId, dstDataOff],
[srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .min }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red(
  cuda::ptx::op_min_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const uint64_t* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red(
  ::cuda::ptx::op_min_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const ::cuda::std::uint64_t* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_min (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.min.u64 [%0, "
      "%1], [%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.u64
[dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .min }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_counted(
  cuda::ptx::op_min_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const uint64_t* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_counted(
  ::cuda::ptx::op_min_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const ::cuda::std::uint64_t* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_min (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys."
      "min.u64 [%0, %1, %2], [%3], %4, [%5];"
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
// fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.u64 [dstLeId,
dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .min }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_multimem(
  cuda::ptx::op_min_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const uint64_t* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_multimem(
  ::cuda::ptx::op_min_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const ::cuda::std::uint64_t* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_min (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.min."
      "u64 [%0, %1], [%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.u64
[dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .min }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_multimem_counted(
  cuda::ptx::op_min_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const uint64_t* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_multimem_counted(
  ::cuda::ptx::op_min_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const ::cuda::std::uint64_t* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_min (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes."
      "relaxed.sys.min.u64 [%0, %1, %2], [%3], %4, [%5];"
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
// fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.u64 [dstLeId, dstDataOff],
[srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .max }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red(
  cuda::ptx::op_max_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const uint64_t* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red(
  ::cuda::ptx::op_max_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const ::cuda::std::uint64_t* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_max (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.max.u64 [%0, "
      "%1], [%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.u64
[dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .max }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_counted(
  cuda::ptx::op_max_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const uint64_t* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_counted(
  ::cuda::ptx::op_max_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const ::cuda::std::uint64_t* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_max (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys."
      "max.u64 [%0, %1, %2], [%3], %4, [%5];"
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
// fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.u64 [dstLeId,
dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .max }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_multimem(
  cuda::ptx::op_max_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const uint64_t* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_multimem(
  ::cuda::ptx::op_max_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const ::cuda::std::uint64_t* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_max (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.max."
      "u64 [%0, %1], [%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.u64
[dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .max }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_multimem_counted(
  cuda::ptx::op_max_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const uint64_t* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_multimem_counted(
  ::cuda::ptx::op_max_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const ::cuda::std::uint64_t* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_max (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes."
      "relaxed.sys.max.u64 [%0, %1, %2], [%3], %4, [%5];"
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
// fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.s64 [dstLeId, dstDataOff],
[srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .min }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red(
  cuda::ptx::op_min_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const int64_t* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red(
  ::cuda::ptx::op_min_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const ::cuda::std::int64_t* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_min (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.min.s64 [%0, "
      "%1], [%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.s64
[dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .min }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_counted(
  cuda::ptx::op_min_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const int64_t* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_counted(
  ::cuda::ptx::op_min_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const ::cuda::std::int64_t* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_min (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys."
      "min.s64 [%0, %1, %2], [%3], %4, [%5];"
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
// fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.s64 [dstLeId,
dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .min }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_multimem(
  cuda::ptx::op_min_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const int64_t* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_multimem(
  ::cuda::ptx::op_min_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const ::cuda::std::int64_t* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_min (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.min."
      "s64 [%0, %1], [%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.s64
[dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .min }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_multimem_counted(
  cuda::ptx::op_min_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const int64_t* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_multimem_counted(
  ::cuda::ptx::op_min_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const ::cuda::std::int64_t* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_min (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes."
      "relaxed.sys.min.s64 [%0, %1, %2], [%3], %4, [%5];"
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
// fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.s64 [dstLeId, dstDataOff],
[srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .max }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red(
  cuda::ptx::op_max_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const int64_t* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red(
  ::cuda::ptx::op_max_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const ::cuda::std::int64_t* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_max (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.max.s64 [%0, "
      "%1], [%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.s64
[dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .max }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_counted(
  cuda::ptx::op_max_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const int64_t* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_counted(
  ::cuda::ptx::op_max_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const ::cuda::std::int64_t* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_max (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys."
      "max.s64 [%0, %1, %2], [%3], %4, [%5];"
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
// fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.s64 [dstLeId,
dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .max }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_multimem(
  cuda::ptx::op_max_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const int64_t* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_multimem(
  ::cuda::ptx::op_max_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const ::cuda::std::int64_t* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_max (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.max."
      "s64 [%0, %1], [%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.s64
[dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .max }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_multimem_counted(
  cuda::ptx::op_max_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const int64_t* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_multimem_counted(
  ::cuda::ptx::op_max_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const ::cuda::std::int64_t* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_max (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes."
      "relaxed.sys.max.s64 [%0, %1, %2], [%3], %4, [%5];"
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
// fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.f16 [dstLeId, dstDataOff],
[srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .min }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red(
  cuda::ptx::op_min_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const __half* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red(
  ::cuda::ptx::op_min_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const __half* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_min (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.min.f16 [%0, "
      "%1], [%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.f16
[dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .min }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_counted(
  cuda::ptx::op_min_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const __half* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_counted(
  ::cuda::ptx::op_min_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const __half* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_min (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys."
      "min.f16 [%0, %1, %2], [%3], %4, [%5];"
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
// fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.f16 [dstLeId,
dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .min }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_multimem(
  cuda::ptx::op_min_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const __half* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_multimem(
  ::cuda::ptx::op_min_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const __half* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_min (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.min."
      "f16 [%0, %1], [%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.f16
[dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .min }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_multimem_counted(
  cuda::ptx::op_min_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const __half* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_multimem_counted(
  ::cuda::ptx::op_min_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const __half* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_min (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes."
      "relaxed.sys.min.f16 [%0, %1, %2], [%3], %4, [%5];"
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
// fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.f16 [dstLeId, dstDataOff],
[srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .max }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red(
  cuda::ptx::op_max_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const __half* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red(
  ::cuda::ptx::op_max_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const __half* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_max (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.max.f16 [%0, "
      "%1], [%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.f16
[dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .max }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_counted(
  cuda::ptx::op_max_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const __half* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_counted(
  ::cuda::ptx::op_max_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const __half* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_max (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys."
      "max.f16 [%0, %1, %2], [%3], %4, [%5];"
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
// fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.f16 [dstLeId,
dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .max }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_multimem(
  cuda::ptx::op_max_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const __half* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_multimem(
  ::cuda::ptx::op_max_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const __half* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_max (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.max."
      "f16 [%0, %1], [%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.f16
[dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .max }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_multimem_counted(
  cuda::ptx::op_max_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const __half* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_multimem_counted(
  ::cuda::ptx::op_max_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const __half* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_max (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes."
      "relaxed.sys.max.f16 [%0, %1, %2], [%3], %4, [%5];"
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
// fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.bf16 [dstLeId, dstDataOff],
[srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .min }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red(
  cuda::ptx::op_min_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const __nv_bfloat16* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red(
  ::cuda::ptx::op_min_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const __nv_bfloat16* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_min (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.min.bf16 [%0, "
      "%1], [%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.bf16
[dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .min }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_counted(
  cuda::ptx::op_min_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const __nv_bfloat16* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_counted(
  ::cuda::ptx::op_min_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const __nv_bfloat16* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_min (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys."
      "min.bf16 [%0, %1, %2], [%3], %4, [%5];"
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
// fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.bf16 [dstLeId,
dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .min }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_multimem(
  cuda::ptx::op_min_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const __nv_bfloat16* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_multimem(
  ::cuda::ptx::op_min_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const __nv_bfloat16* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_min (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.min."
      "bf16 [%0, %1], [%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
//
fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.bf16
[dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .min }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_multimem_counted(
  cuda::ptx::op_min_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const __nv_bfloat16* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_multimem_counted(
  ::cuda::ptx::op_min_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const __nv_bfloat16* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_min (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes."
      "relaxed.sys.min.bf16 [%0, %1, %2], [%3], %4, [%5];"
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
// fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.bf16 [dstLeId, dstDataOff],
[srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .max }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red(
  cuda::ptx::op_max_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const __nv_bfloat16* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red(
  ::cuda::ptx::op_max_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const __nv_bfloat16* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_max (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.max.bf16 [%0, "
      "%1], [%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.bf16
[dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .max }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_counted(
  cuda::ptx::op_max_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const __nv_bfloat16* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_counted(
  ::cuda::ptx::op_max_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const __nv_bfloat16* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_max (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys."
      "max.bf16 [%0, %1, %2], [%3], %4, [%5];"
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
// fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.bf16 [dstLeId,
dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .max }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_multimem(
  cuda::ptx::op_max_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const __nv_bfloat16* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_multimem(
  ::cuda::ptx::op_max_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const __nv_bfloat16* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_max (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.max."
      "bf16 [%0, %1], [%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
//
fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.bf16
[dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .max }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_multimem_counted(
  cuda::ptx::op_max_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const __nv_bfloat16* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_multimem_counted(
  ::cuda::ptx::op_max_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const __nv_bfloat16* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_max (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes."
      "relaxed.sys.max.bf16 [%0, %1, %2], [%3], %4, [%5];"
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
// fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.u32 [dstLeId, dstDataOff],
[srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .add }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red(
  cuda::ptx::op_add_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const uint32_t* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red(
  ::cuda::ptx::op_add_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const ::cuda::std::uint32_t* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_add (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.add.u32 [%0, "
      "%1], [%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.u32
[dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .add }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_counted(
  cuda::ptx::op_add_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const uint32_t* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_counted(
  ::cuda::ptx::op_add_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const ::cuda::std::uint32_t* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_add (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys."
      "add.u32 [%0, %1, %2], [%3], %4, [%5];"
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
// fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.u32 [dstLeId,
dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .add }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_multimem(
  cuda::ptx::op_add_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const uint32_t* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_multimem(
  ::cuda::ptx::op_add_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const ::cuda::std::uint32_t* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_add (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.add."
      "u32 [%0, %1], [%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.u32
[dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .add }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_multimem_counted(
  cuda::ptx::op_add_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const uint32_t* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_multimem_counted(
  ::cuda::ptx::op_add_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const ::cuda::std::uint32_t* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_add (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes."
      "relaxed.sys.add.u32 [%0, %1, %2], [%3], %4, [%5];"
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
// fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.u64 [dstLeId, dstDataOff],
[srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .add }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red(
  cuda::ptx::op_add_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const uint64_t* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red(
  ::cuda::ptx::op_add_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const ::cuda::std::uint64_t* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_add (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.add.u64 [%0, "
      "%1], [%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.u64
[dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .add }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_counted(
  cuda::ptx::op_add_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const uint64_t* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_counted(
  ::cuda::ptx::op_add_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const ::cuda::std::uint64_t* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_add (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys."
      "add.u64 [%0, %1, %2], [%3], %4, [%5];"
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
// fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.u64 [dstLeId,
dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .add }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_multimem(
  cuda::ptx::op_add_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const uint64_t* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_multimem(
  ::cuda::ptx::op_add_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const ::cuda::std::uint64_t* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_add (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.add."
      "u64 [%0, %1], [%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.u64
[dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .add }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_multimem_counted(
  cuda::ptx::op_add_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const uint64_t* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_multimem_counted(
  ::cuda::ptx::op_add_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const ::cuda::std::uint64_t* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_add (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes."
      "relaxed.sys.add.u64 [%0, %1, %2], [%3], %4, [%5];"
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
// fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.f16 [dstLeId, dstDataOff],
[srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .add }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red(
  cuda::ptx::op_add_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const __half* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red(
  ::cuda::ptx::op_add_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const __half* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_add (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.add.f16 [%0, "
      "%1], [%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.f16
[dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .add }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_counted(
  cuda::ptx::op_add_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const __half* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_counted(
  ::cuda::ptx::op_add_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const __half* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_add (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys."
      "add.f16 [%0, %1, %2], [%3], %4, [%5];"
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
// fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.f16 [dstLeId,
dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .add }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_multimem(
  cuda::ptx::op_add_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const __half* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_multimem(
  ::cuda::ptx::op_add_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const __half* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_add (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.add."
      "f16 [%0, %1], [%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.f16
[dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .add }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_multimem_counted(
  cuda::ptx::op_add_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const __half* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_multimem_counted(
  ::cuda::ptx::op_add_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const __half* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_add (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes."
      "relaxed.sys.add.f16 [%0, %1, %2], [%3], %4, [%5];"
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
// fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.bf16 [dstLeId, dstDataOff],
[srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .add }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red(
  cuda::ptx::op_add_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const __nv_bfloat16* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red(
  ::cuda::ptx::op_add_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const __nv_bfloat16* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_add (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.add.bf16 [%0, "
      "%1], [%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.bf16
[dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .add }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_counted(
  cuda::ptx::op_add_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const __nv_bfloat16* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_counted(
  ::cuda::ptx::op_add_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const __nv_bfloat16* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_add (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys."
      "add.bf16 [%0, %1, %2], [%3], %4, [%5];"
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
// fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.bf16 [dstLeId,
dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .add }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_multimem(
  cuda::ptx::op_add_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const __nv_bfloat16* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_multimem(
  ::cuda::ptx::op_add_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const __nv_bfloat16* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_add (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.add."
      "bf16 [%0, %1], [%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
//
fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.bf16
[dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .add }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_multimem_counted(
  cuda::ptx::op_add_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const __nv_bfloat16* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_multimem_counted(
  ::cuda::ptx::op_add_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const __nv_bfloat16* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_add (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes."
      "relaxed.sys.add.bf16 [%0, %1, %2], [%3], %4, [%5];"
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
// fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.f32 [dstLeId, dstDataOff],
[srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .add }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red(
  cuda::ptx::op_add_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const float* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red(
  ::cuda::ptx::op_add_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const float* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_add (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.add.f32 [%0, "
      "%1], [%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.f32
[dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .add }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_counted(
  cuda::ptx::op_add_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const float* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_counted(
  ::cuda::ptx::op_add_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const float* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_add (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys."
      "add.f32 [%0, %1, %2], [%3], %4, [%5];"
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
// fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.f32 [dstLeId,
dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .add }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_multimem(
  cuda::ptx::op_add_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const float* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_multimem(
  ::cuda::ptx::op_add_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const float* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_add (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.add."
      "f32 [%0, %1], [%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.f32
[dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .add }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_multimem_counted(
  cuda::ptx::op_add_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const float* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_multimem_counted(
  ::cuda::ptx::op_add_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const float* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_add (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes."
      "relaxed.sys.add.f32 [%0, %1, %2], [%3], %4, [%5];"
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
// fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.f64 [dstLeId, dstDataOff],
[srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .add }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red(
  cuda::ptx::op_add_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const double* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red(
  ::cuda::ptx::op_add_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const double* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_add (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.add.f64 [%0, "
      "%1], [%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.f64
[dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .add }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_counted(
  cuda::ptx::op_add_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const double* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_counted(
  ::cuda::ptx::op_add_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const double* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_add (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys."
      "add.f64 [%0, %1, %2], [%3], %4, [%5];"
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
// fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.f64 [dstLeId,
dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .add }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_multimem(
  cuda::ptx::op_add_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  const double* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_multimem(
  ::cuda::ptx::op_add_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  const double* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_add (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.add."
      "f64 [%0, %1], [%2], %3, [%4];"
      :
      : "r"(__dstLeId), "l"(__dstDataOff), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 930

/*
// fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.f64
[dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
// .op        = { .add }
// .src       = { .shared::cta }
// .sem       = { .relaxed }
// .scope     = { .sys }
template <typename = void>
__device__ static inline void fabric_try_red_multimem_counted(
  cuda::ptx::op_add_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_sys_t,
  uint32_t dstLeId,
  uint64_t dstDataOff,
  uint64_t dstCounterOff,
  const double* srcMem,
  uint32_t size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 930
template <typename = void>
_CCCL_DEVICE static inline void fabric_try_red_multimem_counted(
  ::cuda::ptx::op_add_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_sys_t,
  ::cuda::std::uint32_t __dstLeId,
  ::cuda::std::uint64_t __dstDataOff,
  ::cuda::std::uint64_t __dstCounterOff,
  const double* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __op == op_add (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_sys (due to parameter type constraint)
  asm("fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes."
      "relaxed.sys.add.f64 [%0, %1, %2], [%3], %4, [%5];"
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

#endif // _CUDA_PTX_GENERATED_FABRIC_TRY_RED_H_
