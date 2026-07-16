// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_MBARRIER_TRY_WAIT_PARITY_H_
#define _CUDA_PTX_GENERATED_MBARRIER_TRY_WAIT_PARITY_H_

/*
// mbarrier.try_wait.parity.shared::cta.b64 waitComplete, [addr], phaseParity; // PTX ISA 78, SM_90
template <typename = void>
__device__ static inline bool mbarrier_try_wait_parity(
  uint64_t* addr,
  const uint32_t& phaseParity);
*/
#if __cccl_ptx_isa >= 780
template <typename = void>
_CCCL_DEVICE static inline bool
mbarrier_try_wait_parity(::cuda::std::uint64_t* __addr, const ::cuda::std::uint32_t& __phaseParity)
{
  ::cuda::std::uint32_t __waitComplete;
  asm("{\n\t"
      ".reg .pred P_OUT; \n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P_OUT, [%1], %2; \n\t"
      "selp.b32 %0, 1, 0, P_OUT; \n"
      "}"
      : "=r"(__waitComplete)
      : "r"(__as_ptr_smem(__addr)), "r"(__phaseParity)
      : "memory");
  return static_cast<bool>(__waitComplete);
}
#endif // __cccl_ptx_isa >= 780

/*
// mbarrier.try_wait.parity.shared::cta.b64 waitComplete, [addr], phaseParity, suspendTimeHint; // PTX ISA 78, SM_90
template <typename = void>
__device__ static inline bool mbarrier_try_wait_parity(
  uint64_t* addr,
  const uint32_t& phaseParity,
  const uint32_t& suspendTimeHint);
*/
#if __cccl_ptx_isa >= 780
template <typename = void>
_CCCL_DEVICE static inline bool mbarrier_try_wait_parity(
  ::cuda::std::uint64_t* __addr,
  const ::cuda::std::uint32_t& __phaseParity,
  const ::cuda::std::uint32_t& __suspendTimeHint)
{
  ::cuda::std::uint32_t __waitComplete;
  asm("{\n\t"
      ".reg .pred P_OUT; \n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P_OUT, [%1], %2, %3; \n\t"
      "selp.b32 %0, 1, 0, P_OUT; \n"
      "}"
      : "=r"(__waitComplete)
      : "r"(__as_ptr_smem(__addr)), "r"(__phaseParity), "r"(__suspendTimeHint)
      : "memory");
  return static_cast<bool>(__waitComplete);
}
#endif // __cccl_ptx_isa >= 780

/*
// mbarrier.try_wait.parity.sem.scope.shared::cta.b64 waitComplete, [addr], phaseParity; // PTX ISA 80, SM_90
// .sem       = { .acquire }
// .scope     = { .cta, .cluster }
template <cuda::ptx::dot_scope Scope>
__device__ static inline bool mbarrier_try_wait_parity(
  cuda::ptx::sem_acquire_t,
  cuda::ptx::scope_t<Scope> scope,
  uint64_t* addr,
  const uint32_t& phaseParity);
*/
#if __cccl_ptx_isa >= 800
template <::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline bool mbarrier_try_wait_parity(
  ::cuda::ptx::sem_acquire_t,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::std::uint64_t* __addr,
  const ::cuda::std::uint32_t& __phaseParity)
{
  // __sem == sem_acquire (due to parameter type constraint)
  static_assert(__scope == scope_cta || __scope == scope_cluster, "");
  ::cuda::std::uint32_t __waitComplete;
  if constexpr (__scope == scope_cta)
  {
    asm("{\n\t"
        ".reg .pred P_OUT; \n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P_OUT, [%1], %2; \n\t"
        "selp.b32 %0, 1, 0, P_OUT; \n"
        "}"
        : "=r"(__waitComplete)
        : "r"(__as_ptr_smem(__addr)), "r"(__phaseParity)
        : "memory");
  }
  else if constexpr (__scope == scope_cluster)
  {
    asm("{\n\t"
        ".reg .pred P_OUT; \n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P_OUT, [%1], %2; \n\t"
        "selp.b32 %0, 1, 0, P_OUT; \n"
        "}"
        : "=r"(__waitComplete)
        : "r"(__as_ptr_smem(__addr)), "r"(__phaseParity)
        : "memory");
  }
  return static_cast<bool>(__waitComplete);
}
#endif // __cccl_ptx_isa >= 800

/*
// mbarrier.try_wait.parity.sem.scope.shared::cta.b64 waitComplete, [addr], phaseParity, suspendTimeHint; // PTX ISA 80,
SM_90
// .sem       = { .acquire }
// .scope     = { .cta, .cluster }
template <cuda::ptx::dot_scope Scope>
__device__ static inline bool mbarrier_try_wait_parity(
  cuda::ptx::sem_acquire_t,
  cuda::ptx::scope_t<Scope> scope,
  uint64_t* addr,
  const uint32_t& phaseParity,
  const uint32_t& suspendTimeHint);
*/
#if __cccl_ptx_isa >= 800
template <::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline bool mbarrier_try_wait_parity(
  ::cuda::ptx::sem_acquire_t,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::std::uint64_t* __addr,
  const ::cuda::std::uint32_t& __phaseParity,
  const ::cuda::std::uint32_t& __suspendTimeHint)
{
  // __sem == sem_acquire (due to parameter type constraint)
  static_assert(__scope == scope_cta || __scope == scope_cluster, "");
  ::cuda::std::uint32_t __waitComplete;
  if constexpr (__scope == scope_cta)
  {
    asm("{\n\t"
        ".reg .pred P_OUT; \n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P_OUT, [%1], %2, %3; \n\t"
        "selp.b32 %0, 1, 0, P_OUT; \n"
        "}"
        : "=r"(__waitComplete)
        : "r"(__as_ptr_smem(__addr)), "r"(__phaseParity), "r"(__suspendTimeHint)
        : "memory");
  }
  else if constexpr (__scope == scope_cluster)
  {
    asm("{\n\t"
        ".reg .pred P_OUT; \n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P_OUT, [%1], %2, %3; \n\t"
        "selp.b32 %0, 1, 0, P_OUT; \n"
        "}"
        : "=r"(__waitComplete)
        : "r"(__as_ptr_smem(__addr)), "r"(__phaseParity), "r"(__suspendTimeHint)
        : "memory");
  }
  return static_cast<bool>(__waitComplete);
}
#endif // __cccl_ptx_isa >= 800

/*
// mbarrier.try_wait.parity.sem.scope.shared::cta.b64 waitComplete, [addr], phaseParity; // PTX ISA 86, SM_90
// .sem       = { .relaxed }
// .scope     = { .cta, .cluster }
template <cuda::ptx::dot_scope Scope>
__device__ static inline bool mbarrier_try_wait_parity(
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_t<Scope> scope,
  uint64_t* addr,
  const uint32_t& phaseParity);
*/
#if __cccl_ptx_isa >= 860
template <::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline bool mbarrier_try_wait_parity(
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::std::uint64_t* __addr,
  const ::cuda::std::uint32_t& __phaseParity)
{
  // __sem == sem_relaxed (due to parameter type constraint)
  static_assert(__scope == scope_cta || __scope == scope_cluster, "");
  ::cuda::std::uint32_t __waitComplete;
  if constexpr (__scope == scope_cta)
  {
    asm("{\n\t"
        ".reg .pred P_OUT; \n\t"
        "mbarrier.try_wait.parity.relaxed.cta.shared::cta.b64 P_OUT, [%1], %2; \n\t"
        "selp.b32 %0, 1, 0, P_OUT; \n"
        "}"
        : "=r"(__waitComplete)
        : "r"(__as_ptr_smem(__addr)), "r"(__phaseParity)
        : "memory");
  }
  else if constexpr (__scope == scope_cluster)
  {
    asm("{\n\t"
        ".reg .pred P_OUT; \n\t"
        "mbarrier.try_wait.parity.relaxed.cluster.shared::cta.b64 P_OUT, [%1], %2; \n\t"
        "selp.b32 %0, 1, 0, P_OUT; \n"
        "}"
        : "=r"(__waitComplete)
        : "r"(__as_ptr_smem(__addr)), "r"(__phaseParity)
        : "memory");
  }
  return static_cast<bool>(__waitComplete);
}
#endif // __cccl_ptx_isa >= 860

/*
// mbarrier.try_wait.parity.sem.scope.shared::cta.b64 waitComplete, [addr], phaseParity, suspendTimeHint; // PTX ISA 86,
SM_90
// .sem       = { .relaxed }
// .scope     = { .cta, .cluster }
template <cuda::ptx::dot_scope Scope>
__device__ static inline bool mbarrier_try_wait_parity(
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_t<Scope> scope,
  uint64_t* addr,
  const uint32_t& phaseParity,
  const uint32_t& suspendTimeHint);
*/
#if __cccl_ptx_isa >= 860
template <::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline bool mbarrier_try_wait_parity(
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::std::uint64_t* __addr,
  const ::cuda::std::uint32_t& __phaseParity,
  const ::cuda::std::uint32_t& __suspendTimeHint)
{
  // __sem == sem_relaxed (due to parameter type constraint)
  static_assert(__scope == scope_cta || __scope == scope_cluster, "");
  ::cuda::std::uint32_t __waitComplete;
  if constexpr (__scope == scope_cta)
  {
    asm("{\n\t"
        ".reg .pred P_OUT; \n\t"
        "mbarrier.try_wait.parity.relaxed.cta.shared::cta.b64 P_OUT, [%1], %2, %3; \n\t"
        "selp.b32 %0, 1, 0, P_OUT; \n"
        "}"
        : "=r"(__waitComplete)
        : "r"(__as_ptr_smem(__addr)), "r"(__phaseParity), "r"(__suspendTimeHint)
        : "memory");
  }
  else if constexpr (__scope == scope_cluster)
  {
    asm("{\n\t"
        ".reg .pred P_OUT; \n\t"
        "mbarrier.try_wait.parity.relaxed.cluster.shared::cta.b64 P_OUT, [%1], %2, %3; \n\t"
        "selp.b32 %0, 1, 0, P_OUT; \n"
        "}"
        : "=r"(__waitComplete)
        : "r"(__as_ptr_smem(__addr)), "r"(__phaseParity), "r"(__suspendTimeHint)
        : "memory");
  }
  return static_cast<bool>(__waitComplete);
}
#endif // __cccl_ptx_isa >= 860

/*
// mbarrier.try_wait.parity.phase_type.sem.scope.shared::cta.b64 waitComplete|isReportSeen, [addr], phaseParity; // PTX
ISA 94, SM_90
// .phase_type = { .phase_type::primary }
// .sem       = { .acquire, .relaxed }
// .scope     = { .cta, .cluster }
template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
__device__ static inline bool mbarrier_try_wait_parity(
  cuda::ptx::mbarrier_phase_primary_t,
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_t<Scope> scope,
  bool& isReportSeen,
  uint64_t* addr,
  uint32_t phaseParity);
*/
#if __cccl_ptx_isa >= 940
template <::cuda::ptx::dot_sem _Sem, ::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline bool mbarrier_try_wait_parity(
  ::cuda::ptx::mbarrier_phase_primary_t,
  ::cuda::ptx::sem_t<_Sem> __sem,
  ::cuda::ptx::scope_t<_Scope> __scope,
  bool& __isReportSeen,
  ::cuda::std::uint64_t* __addr,
  ::cuda::std::uint32_t __phaseParity)
{
  // __phase_type == mbarrier_phase_primary (due to parameter type constraint)
  static_assert(__sem == sem_acquire || __sem == sem_relaxed, "");
  static_assert(__scope == scope_cta || __scope == scope_cluster, "");
  ::cuda::std::uint32_t __waitComplete;
  ::cuda::std::uint32_t __isReportSeen_tmp;
  if constexpr (__sem == sem_acquire && __scope == scope_cta)
  {
    asm("{\n\t"
        ".reg .pred P_OUT_waitComplete; \n\t"
        ".reg .pred P_OUT_isReportSeen; \n\t"
        "mbarrier.try_wait.parity.phase_type::primary.acquire.cta.shared::cta.b64 "
        "P_OUT_waitComplete|P_OUT_isReportSeen, [%2], %3; \n\t"
        "selp.b32 %0, 1, 0, P_OUT_waitComplete; \n\t"
        "selp.b32 %1, 1, 0, P_OUT_isReportSeen; \n"
        "}"
        : "=r"(__waitComplete), "=r"(__isReportSeen_tmp)
        : "r"(__as_ptr_smem(__addr)), "r"(__phaseParity)
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_cluster)
  {
    asm("{\n\t"
        ".reg .pred P_OUT_waitComplete; \n\t"
        ".reg .pred P_OUT_isReportSeen; \n\t"
        "mbarrier.try_wait.parity.phase_type::primary.acquire.cluster.shared::cta.b64 "
        "P_OUT_waitComplete|P_OUT_isReportSeen, [%2], %3; \n\t"
        "selp.b32 %0, 1, 0, P_OUT_waitComplete; \n\t"
        "selp.b32 %1, 1, 0, P_OUT_isReportSeen; \n"
        "}"
        : "=r"(__waitComplete), "=r"(__isReportSeen_tmp)
        : "r"(__as_ptr_smem(__addr)), "r"(__phaseParity)
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_cta)
  {
    asm("{\n\t"
        ".reg .pred P_OUT_waitComplete; \n\t"
        ".reg .pred P_OUT_isReportSeen; \n\t"
        "mbarrier.try_wait.parity.phase_type::primary.relaxed.cta.shared::cta.b64 "
        "P_OUT_waitComplete|P_OUT_isReportSeen, [%2], %3; \n\t"
        "selp.b32 %0, 1, 0, P_OUT_waitComplete; \n\t"
        "selp.b32 %1, 1, 0, P_OUT_isReportSeen; \n"
        "}"
        : "=r"(__waitComplete), "=r"(__isReportSeen_tmp)
        : "r"(__as_ptr_smem(__addr)), "r"(__phaseParity)
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_cluster)
  {
    asm("{\n\t"
        ".reg .pred P_OUT_waitComplete; \n\t"
        ".reg .pred P_OUT_isReportSeen; \n\t"
        "mbarrier.try_wait.parity.phase_type::primary.relaxed.cluster.shared::cta.b64 "
        "P_OUT_waitComplete|P_OUT_isReportSeen, [%2], %3; \n\t"
        "selp.b32 %0, 1, 0, P_OUT_waitComplete; \n\t"
        "selp.b32 %1, 1, 0, P_OUT_isReportSeen; \n"
        "}"
        : "=r"(__waitComplete), "=r"(__isReportSeen_tmp)
        : "r"(__as_ptr_smem(__addr)), "r"(__phaseParity)
        : "memory");
  }
  __isReportSeen = static_cast<bool>(__isReportSeen_tmp);
  return static_cast<bool>(__waitComplete);
}
#endif // __cccl_ptx_isa >= 940

/*
// mbarrier.try_wait.parity.phase_type.sem.scope.shared::cta.b64 waitComplete, [addr], phaseParity; // PTX ISA 94, SM_90
// .phase_type = { .phase_type::conditional }
// .sem       = { .acquire, .relaxed }
// .scope     = { .cta, .cluster }
template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
__device__ static inline bool mbarrier_try_wait_parity(
  cuda::ptx::mbarrier_phase_conditional_t,
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_t<Scope> scope,
  uint64_t* addr,
  uint32_t phaseParity);
*/
#if __cccl_ptx_isa >= 940
template <::cuda::ptx::dot_sem _Sem, ::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline bool mbarrier_try_wait_parity(
  ::cuda::ptx::mbarrier_phase_conditional_t,
  ::cuda::ptx::sem_t<_Sem> __sem,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::std::uint64_t* __addr,
  ::cuda::std::uint32_t __phaseParity)
{
  // __phase_type == mbarrier_phase_conditional (due to parameter type constraint)
  static_assert(__sem == sem_acquire || __sem == sem_relaxed, "");
  static_assert(__scope == scope_cta || __scope == scope_cluster, "");
  ::cuda::std::uint32_t __waitComplete;
  if constexpr (__sem == sem_acquire && __scope == scope_cta)
  {
    asm("{\n\t"
        ".reg .pred P_OUT; \n\t"
        "mbarrier.try_wait.parity.phase_type::conditional.acquire.cta.shared::cta.b64 P_OUT, [%1], %2; \n\t"
        "selp.b32 %0, 1, 0, P_OUT; \n"
        "}"
        : "=r"(__waitComplete)
        : "r"(__as_ptr_smem(__addr)), "r"(__phaseParity)
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_cluster)
  {
    asm("{\n\t"
        ".reg .pred P_OUT; \n\t"
        "mbarrier.try_wait.parity.phase_type::conditional.acquire.cluster.shared::cta.b64 P_OUT, [%1], %2; \n\t"
        "selp.b32 %0, 1, 0, P_OUT; \n"
        "}"
        : "=r"(__waitComplete)
        : "r"(__as_ptr_smem(__addr)), "r"(__phaseParity)
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_cta)
  {
    asm("{\n\t"
        ".reg .pred P_OUT; \n\t"
        "mbarrier.try_wait.parity.phase_type::conditional.relaxed.cta.shared::cta.b64 P_OUT, [%1], %2; \n\t"
        "selp.b32 %0, 1, 0, P_OUT; \n"
        "}"
        : "=r"(__waitComplete)
        : "r"(__as_ptr_smem(__addr)), "r"(__phaseParity)
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_cluster)
  {
    asm("{\n\t"
        ".reg .pred P_OUT; \n\t"
        "mbarrier.try_wait.parity.phase_type::conditional.relaxed.cluster.shared::cta.b64 P_OUT, [%1], %2; \n\t"
        "selp.b32 %0, 1, 0, P_OUT; \n"
        "}"
        : "=r"(__waitComplete)
        : "r"(__as_ptr_smem(__addr)), "r"(__phaseParity)
        : "memory");
  }
  return static_cast<bool>(__waitComplete);
}
#endif // __cccl_ptx_isa >= 940

/*
// mbarrier.try_wait.parity.phase_type.sem.scope.shared::cta.b64 waitComplete|isReportSeen, [addr], phaseParity,
suspendTimeHint; // PTX ISA 94, SM_90
// .phase_type = { .phase_type::primary }
// .sem       = { .acquire, .relaxed }
// .scope     = { .cta, .cluster }
template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
__device__ static inline bool mbarrier_try_wait_parity(
  cuda::ptx::mbarrier_phase_primary_t,
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_t<Scope> scope,
  bool& isReportSeen,
  uint64_t* addr,
  uint32_t phaseParity,
  uint32_t suspendTimeHint);
*/
#if __cccl_ptx_isa >= 940
template <::cuda::ptx::dot_sem _Sem, ::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline bool mbarrier_try_wait_parity(
  ::cuda::ptx::mbarrier_phase_primary_t,
  ::cuda::ptx::sem_t<_Sem> __sem,
  ::cuda::ptx::scope_t<_Scope> __scope,
  bool& __isReportSeen,
  ::cuda::std::uint64_t* __addr,
  ::cuda::std::uint32_t __phaseParity,
  ::cuda::std::uint32_t __suspendTimeHint)
{
  // __phase_type == mbarrier_phase_primary (due to parameter type constraint)
  static_assert(__sem == sem_acquire || __sem == sem_relaxed, "");
  static_assert(__scope == scope_cta || __scope == scope_cluster, "");
  ::cuda::std::uint32_t __waitComplete;
  ::cuda::std::uint32_t __isReportSeen_tmp;
  if constexpr (__sem == sem_acquire && __scope == scope_cta)
  {
    asm("{\n\t"
        ".reg .pred P_OUT_waitComplete; \n\t"
        ".reg .pred P_OUT_isReportSeen; \n\t"
        "mbarrier.try_wait.parity.phase_type::primary.acquire.cta.shared::cta.b64 "
        "P_OUT_waitComplete|P_OUT_isReportSeen, [%2], %3, %4; \n\t"
        "selp.b32 %0, 1, 0, P_OUT_waitComplete; \n\t"
        "selp.b32 %1, 1, 0, P_OUT_isReportSeen; \n"
        "}"
        : "=r"(__waitComplete), "=r"(__isReportSeen_tmp)
        : "r"(__as_ptr_smem(__addr)), "r"(__phaseParity), "r"(__suspendTimeHint)
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_cluster)
  {
    asm("{\n\t"
        ".reg .pred P_OUT_waitComplete; \n\t"
        ".reg .pred P_OUT_isReportSeen; \n\t"
        "mbarrier.try_wait.parity.phase_type::primary.acquire.cluster.shared::cta.b64 "
        "P_OUT_waitComplete|P_OUT_isReportSeen, [%2], %3, %4; \n\t"
        "selp.b32 %0, 1, 0, P_OUT_waitComplete; \n\t"
        "selp.b32 %1, 1, 0, P_OUT_isReportSeen; \n"
        "}"
        : "=r"(__waitComplete), "=r"(__isReportSeen_tmp)
        : "r"(__as_ptr_smem(__addr)), "r"(__phaseParity), "r"(__suspendTimeHint)
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_cta)
  {
    asm("{\n\t"
        ".reg .pred P_OUT_waitComplete; \n\t"
        ".reg .pred P_OUT_isReportSeen; \n\t"
        "mbarrier.try_wait.parity.phase_type::primary.relaxed.cta.shared::cta.b64 "
        "P_OUT_waitComplete|P_OUT_isReportSeen, [%2], %3, %4; \n\t"
        "selp.b32 %0, 1, 0, P_OUT_waitComplete; \n\t"
        "selp.b32 %1, 1, 0, P_OUT_isReportSeen; \n"
        "}"
        : "=r"(__waitComplete), "=r"(__isReportSeen_tmp)
        : "r"(__as_ptr_smem(__addr)), "r"(__phaseParity), "r"(__suspendTimeHint)
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_cluster)
  {
    asm("{\n\t"
        ".reg .pred P_OUT_waitComplete; \n\t"
        ".reg .pred P_OUT_isReportSeen; \n\t"
        "mbarrier.try_wait.parity.phase_type::primary.relaxed.cluster.shared::cta.b64 "
        "P_OUT_waitComplete|P_OUT_isReportSeen, [%2], %3, %4; \n\t"
        "selp.b32 %0, 1, 0, P_OUT_waitComplete; \n\t"
        "selp.b32 %1, 1, 0, P_OUT_isReportSeen; \n"
        "}"
        : "=r"(__waitComplete), "=r"(__isReportSeen_tmp)
        : "r"(__as_ptr_smem(__addr)), "r"(__phaseParity), "r"(__suspendTimeHint)
        : "memory");
  }
  __isReportSeen = static_cast<bool>(__isReportSeen_tmp);
  return static_cast<bool>(__waitComplete);
}
#endif // __cccl_ptx_isa >= 940

/*
// mbarrier.try_wait.parity.phase_type.sem.scope.shared::cta.b64 waitComplete, [addr], phaseParity, suspendTimeHint; //
PTX ISA 94, SM_90
// .phase_type = { .phase_type::conditional }
// .sem       = { .acquire, .relaxed }
// .scope     = { .cta, .cluster }
template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
__device__ static inline bool mbarrier_try_wait_parity(
  cuda::ptx::mbarrier_phase_conditional_t,
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_t<Scope> scope,
  uint64_t* addr,
  uint32_t phaseParity,
  uint32_t suspendTimeHint);
*/
#if __cccl_ptx_isa >= 940
template <::cuda::ptx::dot_sem _Sem, ::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline bool mbarrier_try_wait_parity(
  ::cuda::ptx::mbarrier_phase_conditional_t,
  ::cuda::ptx::sem_t<_Sem> __sem,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::std::uint64_t* __addr,
  ::cuda::std::uint32_t __phaseParity,
  ::cuda::std::uint32_t __suspendTimeHint)
{
  // __phase_type == mbarrier_phase_conditional (due to parameter type constraint)
  static_assert(__sem == sem_acquire || __sem == sem_relaxed, "");
  static_assert(__scope == scope_cta || __scope == scope_cluster, "");
  ::cuda::std::uint32_t __waitComplete;
  if constexpr (__sem == sem_acquire && __scope == scope_cta)
  {
    asm("{\n\t"
        ".reg .pred P_OUT; \n\t"
        "mbarrier.try_wait.parity.phase_type::conditional.acquire.cta.shared::cta.b64 P_OUT, [%1], %2, %3; \n\t"
        "selp.b32 %0, 1, 0, P_OUT; \n"
        "}"
        : "=r"(__waitComplete)
        : "r"(__as_ptr_smem(__addr)), "r"(__phaseParity), "r"(__suspendTimeHint)
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_cluster)
  {
    asm("{\n\t"
        ".reg .pred P_OUT; \n\t"
        "mbarrier.try_wait.parity.phase_type::conditional.acquire.cluster.shared::cta.b64 P_OUT, [%1], %2, %3; \n\t"
        "selp.b32 %0, 1, 0, P_OUT; \n"
        "}"
        : "=r"(__waitComplete)
        : "r"(__as_ptr_smem(__addr)), "r"(__phaseParity), "r"(__suspendTimeHint)
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_cta)
  {
    asm("{\n\t"
        ".reg .pred P_OUT; \n\t"
        "mbarrier.try_wait.parity.phase_type::conditional.relaxed.cta.shared::cta.b64 P_OUT, [%1], %2, %3; \n\t"
        "selp.b32 %0, 1, 0, P_OUT; \n"
        "}"
        : "=r"(__waitComplete)
        : "r"(__as_ptr_smem(__addr)), "r"(__phaseParity), "r"(__suspendTimeHint)
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_cluster)
  {
    asm("{\n\t"
        ".reg .pred P_OUT; \n\t"
        "mbarrier.try_wait.parity.phase_type::conditional.relaxed.cluster.shared::cta.b64 P_OUT, [%1], %2, %3; \n\t"
        "selp.b32 %0, 1, 0, P_OUT; \n"
        "}"
        : "=r"(__waitComplete)
        : "r"(__as_ptr_smem(__addr)), "r"(__phaseParity), "r"(__suspendTimeHint)
        : "memory");
  }
  return static_cast<bool>(__waitComplete);
}
#endif // __cccl_ptx_isa >= 940

#endif // _CUDA_PTX_GENERATED_MBARRIER_TRY_WAIT_PARITY_H_
