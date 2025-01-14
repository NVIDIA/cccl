// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_MBARRIER_ARRIVE_EXPECT_TX_H_
#define _CUDA_PTX_GENERATED_MBARRIER_ARRIVE_EXPECT_TX_H_

/*
// mbarrier.arrive.expect_tx{.sem}{.scope}{.space}.b64 state, [addr], tx_count; // 8.  PTX ISA 80, SM_90
// .sem       = { .release }
// .scope     = { .cta, .cluster }
// .space     = { .shared::cta }
template <cuda::ptx::dot_scope Scope>
__device__ static inline uint64_t mbarrier_arrive_expect_tx(
  cuda::ptx::sem_release_t,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::space_shared_t,
  uint64_t* addr,
  const uint32_t& tx_count);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_mbarrier_arrive_expect_tx_is_not_supported_before_SM_90__();
template <dot_scope _Scope>
_CCCL_DEVICE static inline _CUDA_VSTD::uint64_t mbarrier_arrive_expect_tx(
  sem_release_t,
  scope_t<_Scope> __scope,
  space_shared_t,
  _CUDA_VSTD::uint64_t* __addr,
  const _CUDA_VSTD::uint32_t& __tx_count)
{
  // __sem == sem_release (due to parameter type constraint)
  static_assert(__scope == scope_cta || __scope == scope_cluster, "");
  // __space == space_shared (due to parameter type constraint)
  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_90,
    (
      _CUDA_VSTD::uint64_t __state; _CCCL_IF_CONSTEXPR (__scope == scope_cta) {
        asm("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 %0, [%1], %2; // 8. "
            : "=l"(__state)
            : "r"(__as_ptr_smem(__addr)), "r"(__tx_count)
            : "memory");
      } else _CCCL_IF_CONSTEXPR (__scope == scope_cluster) {
        asm("mbarrier.arrive.expect_tx.release.cluster.shared::cta.b64 %0, [%1], %2; // 8. "
            : "=l"(__state)
            : "r"(__as_ptr_smem(__addr)), "r"(__tx_count)
            : "memory");
      } return __state;),
    (
      // Unsupported architectures will have a linker error with a semi-decent error message
      __cuda_ptx_mbarrier_arrive_expect_tx_is_not_supported_before_SM_90__(); return 0;));
}
#endif // __cccl_ptx_isa >= 800

/*
// mbarrier.arrive.expect_tx{.sem}{.scope}{.space}.b64   _, [addr], tx_count; // 9.  PTX ISA 80, SM_90
// .sem       = { .release }
// .scope     = { .cluster }
// .space     = { .shared::cluster }
template <typename = void>
__device__ static inline void mbarrier_arrive_expect_tx(
  cuda::ptx::sem_release_t,
  cuda::ptx::scope_cluster_t,
  cuda::ptx::space_cluster_t,
  uint64_t* addr,
  const uint32_t& tx_count);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_mbarrier_arrive_expect_tx_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void mbarrier_arrive_expect_tx(
  sem_release_t, scope_cluster_t, space_cluster_t, _CUDA_VSTD::uint64_t* __addr, const _CUDA_VSTD::uint32_t& __tx_count)
{
  // __sem == sem_release (due to parameter type constraint)
  // __scope == scope_cluster (due to parameter type constraint)
  // __space == space_cluster (due to parameter type constraint)
  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_90,
    (asm("mbarrier.arrive.expect_tx.release.cluster.shared::cluster.b64   _, [%0], %1; // 9. " : : "r"(
           __as_ptr_remote_dsmem(__addr)),
         "r"(__tx_count) : "memory");),
    (
      // Unsupported architectures will have a linker error with a semi-decent error message
      __cuda_ptx_mbarrier_arrive_expect_tx_is_not_supported_before_SM_90__();));
}
#endif // __cccl_ptx_isa >= 800

#endif // _CUDA_PTX_GENERATED_MBARRIER_ARRIVE_EXPECT_TX_H_
