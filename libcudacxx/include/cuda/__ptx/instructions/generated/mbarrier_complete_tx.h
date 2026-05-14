// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_MBARRIER_COMPLETE_TX_H_
#define _CUDA_PTX_GENERATED_MBARRIER_COMPLETE_TX_H_

/*
// mbarrier.complete_tx.sem.scope.space.b64 [addr], txCount; // PTX ISA 80, SM_90
// .sem       = { .relaxed }
// .scope     = { .cta, .cluster }
// .space     = { .shared::cta }
template <cuda::ptx::dot_scope Scope>
__device__ static inline void mbarrier_complete_tx(
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::space_shared_t,
  uint64_t* addr,
  uint32_t txCount);
*/
#if __cccl_ptx_isa >= 800
template <::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline void mbarrier_complete_tx(
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::ptx::space_shared_t,
  ::cuda::std::uint64_t* __addr,
  ::cuda::std::uint32_t __txCount)
{
  // __sem == sem_relaxed (due to parameter type constraint)
  static_assert(__scope == scope_cta || __scope == scope_cluster, "");
  // __space == space_shared (due to parameter type constraint)
  if constexpr (__scope == scope_cta)
  {
    asm("mbarrier.complete_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
        :
        : "r"(__as_ptr_smem(__addr)), "r"(__txCount)
        : "memory");
  }
  else if constexpr (__scope == scope_cluster)
  {
    asm("mbarrier.complete_tx.relaxed.cluster.shared::cta.b64 [%0], %1;"
        :
        : "r"(__as_ptr_smem(__addr)), "r"(__txCount)
        : "memory");
  }
}
#endif // __cccl_ptx_isa >= 800

/*
// mbarrier.complete_tx.sem.scope.space.b64 [addr], txCount; // PTX ISA 80, SM_90
// .sem       = { .relaxed }
// .scope     = { .cta, .cluster }
// .space     = { .shared::cluster }
template <cuda::ptx::dot_scope Scope>
__device__ static inline void mbarrier_complete_tx(
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::space_cluster_t,
  uint64_t* addr,
  uint32_t txCount);
*/
#if __cccl_ptx_isa >= 800
template <::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline void mbarrier_complete_tx(
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::ptx::space_cluster_t,
  ::cuda::std::uint64_t* __addr,
  ::cuda::std::uint32_t __txCount)
{
  // __sem == sem_relaxed (due to parameter type constraint)
  static_assert(__scope == scope_cta || __scope == scope_cluster, "");
  // __space == space_cluster (due to parameter type constraint)
  if constexpr (__scope == scope_cta)
  {
    asm("mbarrier.complete_tx.relaxed.cta.shared::cluster.b64 [%0], %1;"
        :
        : "r"(__as_ptr_dsmem(__addr)), "r"(__txCount)
        : "memory");
  }
  else if constexpr (__scope == scope_cluster)
  {
    asm("mbarrier.complete_tx.relaxed.cluster.shared::cluster.b64 [%0], %1;"
        :
        : "r"(__as_ptr_dsmem(__addr)), "r"(__txCount)
        : "memory");
  }
}
#endif // __cccl_ptx_isa >= 800

/*
// mbarrier.complete_tx.sem.scope.space.multicast::cluster::32b.b64 [addr], txCount, ctaMask; // PTX ISA 94, SM_107a,
SM_107f
// .sem       = { .relaxed }
// .scope     = { .cta, .cluster }
// .space     = { .shared::cluster }
template <cuda::ptx::dot_scope Scope>
__device__ static inline void mbarrier_complete_tx_multicast_32b(
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::space_cluster_t,
  uint64_t* addr,
  uint32_t txCount,
  uint32_t ctaMask);
*/
#if __cccl_ptx_isa >= 940
template <::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline void mbarrier_complete_tx_multicast_32b(
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::ptx::space_cluster_t,
  ::cuda::std::uint64_t* __addr,
  ::cuda::std::uint32_t __txCount,
  ::cuda::std::uint32_t __ctaMask)
{
  // __sem == sem_relaxed (due to parameter type constraint)
  static_assert(__scope == scope_cta || __scope == scope_cluster, "");
  // __space == space_cluster (due to parameter type constraint)
  if constexpr (__scope == scope_cta)
  {
    asm("mbarrier.complete_tx.relaxed.cta.shared::cluster.multicast::cluster::32b.b64 [%0], %1, %2;"
        :
        : "r"(__as_ptr_dsmem(__addr)), "r"(__txCount), "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__scope == scope_cluster)
  {
    asm("mbarrier.complete_tx.relaxed.cluster.shared::cluster.multicast::cluster::32b.b64 [%0], %1, %2;"
        :
        : "r"(__as_ptr_dsmem(__addr)), "r"(__txCount), "r"(__ctaMask)
        : "memory");
  }
}
#endif // __cccl_ptx_isa >= 940

#endif // _CUDA_PTX_GENERATED_MBARRIER_COMPLETE_TX_H_
