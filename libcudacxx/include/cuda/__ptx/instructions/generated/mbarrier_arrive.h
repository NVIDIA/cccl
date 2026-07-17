// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_MBARRIER_ARRIVE_H_
#define _CUDA_PTX_GENERATED_MBARRIER_ARRIVE_H_

/*
// mbarrier.arrive.shared.b64                                  state,  [addr];           // 1.  PTX ISA 70, SM_80
template <typename = void>
__device__ static inline uint64_t mbarrier_arrive(
  uint64_t* addr);
*/
#if __cccl_ptx_isa >= 700
template <typename = void>
_CCCL_DEVICE static inline ::cuda::std::uint64_t mbarrier_arrive(::cuda::std::uint64_t* __addr)
{
  ::cuda::std::uint64_t __state;
  asm("mbarrier.arrive.shared.b64                                  %0,  [%1];           // 1. "
      : "=l"(__state)
      : "r"(__as_ptr_smem(__addr))
      : "memory");
  return __state;
}
#endif // __cccl_ptx_isa >= 700

/*
// mbarrier.arrive.shared::cta.b64                             state,  [addr], count;    // 2.  PTX ISA 78, SM_90
template <typename = void>
__device__ static inline uint64_t mbarrier_arrive(
  uint64_t* addr,
  const uint32_t& count);
*/
#if __cccl_ptx_isa >= 780
template <typename = void>
_CCCL_DEVICE static inline ::cuda::std::uint64_t
mbarrier_arrive(::cuda::std::uint64_t* __addr, const ::cuda::std::uint32_t& __count)
{
  ::cuda::std::uint64_t __state;
  asm("mbarrier.arrive.shared::cta.b64                             %0,  [%1], %2;    // 2. "
      : "=l"(__state)
      : "r"(__as_ptr_smem(__addr)), "r"(__count)
      : "memory");
  return __state;
}
#endif // __cccl_ptx_isa >= 780

/*
// mbarrier.arrive.sem.scope.space.b64                   state,  [addr];           // 3a.  PTX ISA 80, SM_90
// .sem       = { .release }
// .scope     = { .cta, .cluster }
// .space     = { .shared::cta }
template <cuda::ptx::dot_scope Scope>
__device__ static inline uint64_t mbarrier_arrive(
  cuda::ptx::sem_release_t,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::space_shared_t,
  uint64_t* addr);
*/
#if __cccl_ptx_isa >= 800
template <::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline ::cuda::std::uint64_t mbarrier_arrive(
  ::cuda::ptx::sem_release_t,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::ptx::space_shared_t,
  ::cuda::std::uint64_t* __addr)
{
  // __sem == sem_release (due to parameter type constraint)
  static_assert(__scope == scope_cta || __scope == scope_cluster, "");
  // __space == space_shared (due to parameter type constraint)
  ::cuda::std::uint64_t __state;
  if constexpr (__scope == scope_cta)
  {
    asm("mbarrier.arrive.release.cta.shared::cta.b64                   %0,  [%1];           // 3a. "
        : "=l"(__state)
        : "r"(__as_ptr_smem(__addr))
        : "memory");
  }
  else if constexpr (__scope == scope_cluster)
  {
    asm("mbarrier.arrive.release.cluster.shared::cta.b64                   %0,  [%1];           // 3a. "
        : "=l"(__state)
        : "r"(__as_ptr_smem(__addr))
        : "memory");
  }
  return __state;
}
#endif // __cccl_ptx_isa >= 800

/*
// mbarrier.arrive.sem.scope.space.b64                   state,  [addr], count;    // 3b.  PTX ISA 80, SM_90
// .sem       = { .release }
// .scope     = { .cta, .cluster }
// .space     = { .shared::cta }
template <cuda::ptx::dot_scope Scope>
__device__ static inline uint64_t mbarrier_arrive(
  cuda::ptx::sem_release_t,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::space_shared_t,
  uint64_t* addr,
  const uint32_t& count);
*/
#if __cccl_ptx_isa >= 800
template <::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline ::cuda::std::uint64_t mbarrier_arrive(
  ::cuda::ptx::sem_release_t,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::ptx::space_shared_t,
  ::cuda::std::uint64_t* __addr,
  const ::cuda::std::uint32_t& __count)
{
  // __sem == sem_release (due to parameter type constraint)
  static_assert(__scope == scope_cta || __scope == scope_cluster, "");
  // __space == space_shared (due to parameter type constraint)
  ::cuda::std::uint64_t __state;
  if constexpr (__scope == scope_cta)
  {
    asm("mbarrier.arrive.release.cta.shared::cta.b64                   %0,  [%1], %2;    // 3b. "
        : "=l"(__state)
        : "r"(__as_ptr_smem(__addr)), "r"(__count)
        : "memory");
  }
  else if constexpr (__scope == scope_cluster)
  {
    asm("mbarrier.arrive.release.cluster.shared::cta.b64                   %0,  [%1], %2;    // 3b. "
        : "=l"(__state)
        : "r"(__as_ptr_smem(__addr)), "r"(__count)
        : "memory");
  }
  return __state;
}
#endif // __cccl_ptx_isa >= 800

/*
// mbarrier.arrive.sem.scope.space.b64                   _, [addr];                // 4a.  PTX ISA 80, SM_90
// .sem       = { .release }
// .scope     = { .cluster }
// .space     = { .shared::cluster }
template <typename = void>
__device__ static inline void mbarrier_arrive(
  cuda::ptx::sem_release_t,
  cuda::ptx::scope_cluster_t,
  cuda::ptx::space_cluster_t,
  uint64_t* addr);
*/
#if __cccl_ptx_isa >= 800
template <typename = void>
_CCCL_DEVICE static inline void mbarrier_arrive(
  ::cuda::ptx::sem_release_t, ::cuda::ptx::scope_cluster_t, ::cuda::ptx::space_cluster_t, ::cuda::std::uint64_t* __addr)
{
  // __sem == sem_release (due to parameter type constraint)
  // __scope == scope_cluster (due to parameter type constraint)
  // __space == space_cluster (due to parameter type constraint)
  asm("mbarrier.arrive.release.cluster.shared::cluster.b64                   _, [%0];                // 4a. "
      :
      : "r"(__as_ptr_remote_dsmem(__addr))
      : "memory");
}
#endif // __cccl_ptx_isa >= 800

/*
// mbarrier.arrive.sem.scope.space.b64                   _, [addr], count;         // 4b.  PTX ISA 80, SM_90
// .sem       = { .release }
// .scope     = { .cluster }
// .space     = { .shared::cluster }
template <typename = void>
__device__ static inline void mbarrier_arrive(
  cuda::ptx::sem_release_t,
  cuda::ptx::scope_cluster_t,
  cuda::ptx::space_cluster_t,
  uint64_t* addr,
  const uint32_t& count);
*/
#if __cccl_ptx_isa >= 800
template <typename = void>
_CCCL_DEVICE static inline void mbarrier_arrive(
  ::cuda::ptx::sem_release_t,
  ::cuda::ptx::scope_cluster_t,
  ::cuda::ptx::space_cluster_t,
  ::cuda::std::uint64_t* __addr,
  const ::cuda::std::uint32_t& __count)
{
  // __sem == sem_release (due to parameter type constraint)
  // __scope == scope_cluster (due to parameter type constraint)
  // __space == space_cluster (due to parameter type constraint)
  asm("mbarrier.arrive.release.cluster.shared::cluster.b64                   _, [%0], %1;         // 4b. "
      :
      : "r"(__as_ptr_remote_dsmem(__addr)), "r"(__count)
      : "memory");
}
#endif // __cccl_ptx_isa >= 800

/*
// mbarrier.arrive.sem.scope.space.b64 state, [addr], count; // PTX ISA 86, SM_90
// .sem       = { .relaxed }
// .scope     = { .cta, .cluster }
// .space     = { .shared::cta }
template <cuda::ptx::dot_scope Scope>
__device__ static inline uint64_t mbarrier_arrive(
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::space_shared_t,
  uint64_t* addr,
  const uint32_t& count);
*/
#if __cccl_ptx_isa >= 860
template <::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline ::cuda::std::uint64_t mbarrier_arrive(
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::ptx::space_shared_t,
  ::cuda::std::uint64_t* __addr,
  const ::cuda::std::uint32_t& __count)
{
  // __sem == sem_relaxed (due to parameter type constraint)
  static_assert(__scope == scope_cta || __scope == scope_cluster, "");
  // __space == space_shared (due to parameter type constraint)
  ::cuda::std::uint64_t __state;
  if constexpr (__scope == scope_cta)
  {
    asm("mbarrier.arrive.relaxed.cta.shared::cta.b64 %0, [%1], %2;"
        : "=l"(__state)
        : "r"(__as_ptr_smem(__addr)), "r"(__count)
        : "memory");
  }
  else if constexpr (__scope == scope_cluster)
  {
    asm("mbarrier.arrive.relaxed.cluster.shared::cta.b64 %0, [%1], %2;"
        : "=l"(__state)
        : "r"(__as_ptr_smem(__addr)), "r"(__count)
        : "memory");
  }
  return __state;
}
#endif // __cccl_ptx_isa >= 860

/*
// mbarrier.arrive.sem.scope.space.b64 state, [addr]; // PTX ISA 86, SM_90
// .sem       = { .relaxed }
// .scope     = { .cta, .cluster }
// .space     = { .shared::cta }
template <cuda::ptx::dot_scope Scope>
__device__ static inline uint64_t mbarrier_arrive(
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::space_shared_t,
  uint64_t* addr);
*/
#if __cccl_ptx_isa >= 860
template <::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline ::cuda::std::uint64_t mbarrier_arrive(
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::ptx::space_shared_t,
  ::cuda::std::uint64_t* __addr)
{
  // __sem == sem_relaxed (due to parameter type constraint)
  static_assert(__scope == scope_cta || __scope == scope_cluster, "");
  // __space == space_shared (due to parameter type constraint)
  ::cuda::std::uint64_t __state;
  if constexpr (__scope == scope_cta)
  {
    asm("mbarrier.arrive.relaxed.cta.shared::cta.b64 %0, [%1];"
        : "=l"(__state)
        : "r"(__as_ptr_smem(__addr))
        : "memory");
  }
  else if constexpr (__scope == scope_cluster)
  {
    asm("mbarrier.arrive.relaxed.cluster.shared::cta.b64 %0, [%1];"
        : "=l"(__state)
        : "r"(__as_ptr_smem(__addr))
        : "memory");
  }
  return __state;
}
#endif // __cccl_ptx_isa >= 860

/*
// mbarrier.arrive.sem.scope.space.b64 _, [addr], count; // PTX ISA 86, SM_90
// .sem       = { .relaxed }
// .scope     = { .cluster }
// .space     = { .shared::cluster }
template <typename = void>
__device__ static inline void mbarrier_arrive(
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_cluster_t,
  cuda::ptx::space_cluster_t,
  uint64_t* addr,
  const uint32_t& count);
*/
#if __cccl_ptx_isa >= 860
template <typename = void>
_CCCL_DEVICE static inline void mbarrier_arrive(
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_cluster_t,
  ::cuda::ptx::space_cluster_t,
  ::cuda::std::uint64_t* __addr,
  const ::cuda::std::uint32_t& __count)
{
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_cluster (due to parameter type constraint)
  // __space == space_cluster (due to parameter type constraint)
  asm("mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [%0], %1;"
      :
      : "r"(__as_ptr_smem(__addr)), "r"(__count)
      : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// mbarrier.arrive.sem.scope.space.b64 _, [addr]; // PTX ISA 86, SM_90
// .sem       = { .relaxed }
// .scope     = { .cluster }
// .space     = { .shared::cluster }
template <typename = void>
__device__ static inline void mbarrier_arrive(
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_cluster_t,
  cuda::ptx::space_cluster_t,
  uint64_t* addr);
*/
#if __cccl_ptx_isa >= 860
template <typename = void>
_CCCL_DEVICE static inline void mbarrier_arrive(
  ::cuda::ptx::sem_relaxed_t, ::cuda::ptx::scope_cluster_t, ::cuda::ptx::space_cluster_t, ::cuda::std::uint64_t* __addr)
{
  // __sem == sem_relaxed (due to parameter type constraint)
  // __scope == scope_cluster (due to parameter type constraint)
  // __space == space_cluster (due to parameter type constraint)
  asm("mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [%0];" : : "r"(__as_ptr_smem(__addr)) : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// mbarrier.arrive.sem.scope.space.multicast::cluster::32b.b64 _, [addr], count, ctaMask; // PTX ISA 94, SM_107a,
SM_107f
// .sem       = { .release, .relaxed }
// .scope     = { .cluster }
// .space     = { .shared::cluster }
template <cuda::ptx::dot_sem Sem>
__device__ static inline void mbarrier_arrive_multicast_32b(
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_cluster_t,
  cuda::ptx::space_cluster_t,
  uint64_t* addr,
  const uint32_t& count,
  uint32_t ctaMask);
*/
#if __cccl_ptx_isa >= 940
template <::cuda::ptx::dot_sem _Sem>
_CCCL_DEVICE static inline void mbarrier_arrive_multicast_32b(
  ::cuda::ptx::sem_t<_Sem> __sem,
  ::cuda::ptx::scope_cluster_t,
  ::cuda::ptx::space_cluster_t,
  ::cuda::std::uint64_t* __addr,
  const ::cuda::std::uint32_t& __count,
  ::cuda::std::uint32_t __ctaMask)
{
  static_assert(__sem == sem_release || __sem == sem_relaxed, "");
  // __scope == scope_cluster (due to parameter type constraint)
  // __space == space_cluster (due to parameter type constraint)
  if constexpr (__sem == sem_release)
  {
    asm("mbarrier.arrive.release.cluster.shared::cluster.multicast::cluster::32b.b64 _, [%0], %1, %2;"
        :
        : "r"(__as_ptr_smem(__addr)), "r"(__count), "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed)
  {
    asm("mbarrier.arrive.relaxed.cluster.shared::cluster.multicast::cluster::32b.b64 _, [%0], %1, %2;"
        :
        : "r"(__as_ptr_smem(__addr)), "r"(__count), "r"(__ctaMask)
        : "memory");
  }
}
#endif // __cccl_ptx_isa >= 940

/*
// mbarrier.arrive.expect_tx.sem.scope.space.multicast::cluster::32b.b64 _, [addr], txCount, ctaMask; // PTX ISA 94,
SM_107a, SM_107f
// .sem       = { .release, .relaxed }
// .scope     = { .cluster }
// .space     = { .shared::cluster }
template <cuda::ptx::dot_sem Sem>
__device__ static inline void mbarrier_arrive_expect_tx_multicast_32b(
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_cluster_t,
  cuda::ptx::space_cluster_t,
  uint64_t* addr,
  const uint32_t& txCount,
  uint32_t ctaMask);
*/
#if __cccl_ptx_isa >= 940
template <::cuda::ptx::dot_sem _Sem>
_CCCL_DEVICE static inline void mbarrier_arrive_expect_tx_multicast_32b(
  ::cuda::ptx::sem_t<_Sem> __sem,
  ::cuda::ptx::scope_cluster_t,
  ::cuda::ptx::space_cluster_t,
  ::cuda::std::uint64_t* __addr,
  const ::cuda::std::uint32_t& __txCount,
  ::cuda::std::uint32_t __ctaMask)
{
  static_assert(__sem == sem_release || __sem == sem_relaxed, "");
  // __scope == scope_cluster (due to parameter type constraint)
  // __space == space_cluster (due to parameter type constraint)
  if constexpr (__sem == sem_release)
  {
    asm("mbarrier.arrive.expect_tx.release.cluster.shared::cluster.multicast::cluster::32b.b64 _, [%0], %1, %2;"
        :
        : "r"(__as_ptr_dsmem(__addr)), "r"(__txCount), "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed)
  {
    asm("mbarrier.arrive.expect_tx.relaxed.cluster.shared::cluster.multicast::cluster::32b.b64 _, [%0], %1, %2;"
        :
        : "r"(__as_ptr_dsmem(__addr)), "r"(__txCount), "r"(__ctaMask)
        : "memory");
  }
}
#endif // __cccl_ptx_isa >= 940

#endif // _CUDA_PTX_GENERATED_MBARRIER_ARRIVE_H_
