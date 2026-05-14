// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_TCGEN05_COMMIT_H_
#define _CUDA_PTX_GENERATED_TCGEN05_COMMIT_H_

/*
// tcgen05.commit.cta_group.mbarrier::arrive::one.shared::cluster.b64 [smem_bar]; // PTX ISA 86, SM_100a, SM_100f,
SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_commit(
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 860
template <::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void
tcgen05_commit(::cuda::ptx::cta_group_t<_Cta_Group> __cta_group, ::cuda::std::uint64_t* __smem_bar)
{
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
  if constexpr (__cta_group == cta_group_1)
  {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                 :
                 : "r"(__as_ptr_dsmem(__smem_bar))
                 : "memory");
  }
  else if constexpr (__cta_group == cta_group_2)
  {
    asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                 :
                 : "r"(__as_ptr_dsmem(__smem_bar))
                 : "memory");
  }
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.commit.cta_group.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [smem_bar], ctaMask; // PTX ISA
86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_commit_multicast(
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint64_t* smem_bar,
  uint16_t ctaMask);
*/
#if __cccl_ptx_isa >= 860
template <::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_commit_multicast(
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group, ::cuda::std::uint64_t* __smem_bar, ::cuda::std::uint16_t __ctaMask)
{
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
  if constexpr (__cta_group == cta_group_1)
  {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
                 :
                 : "r"(__as_ptr_dsmem(__smem_bar)), "h"(__ctaMask)
                 : "memory");
  }
  else if constexpr (__cta_group == cta_group_2)
  {
    asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
                 :
                 : "r"(__as_ptr_dsmem(__smem_bar)), "h"(__ctaMask)
                 : "memory");
  }
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.commit.cta_group.mbarrier::arrive::one.shared::cluster.multicast::cluster::32b.b64 [smem_bar], ctaMask; //
PTX ISA 94, SM_107a, SM_107f
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_commit_multicast_32b(
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint64_t* smem_bar,
  uint32_t ctaMask);
*/
#if __cccl_ptx_isa >= 940
template <::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_commit_multicast_32b(
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group, ::cuda::std::uint64_t* __smem_bar, ::cuda::std::uint32_t __ctaMask)
{
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
  if constexpr (__cta_group == cta_group_1)
  {
    asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.multicast::cluster::32b.b64 [%0], %1;"
      :
      : "r"(__as_ptr_dsmem(__smem_bar)), "r"(__ctaMask)
      : "memory");
  }
  else if constexpr (__cta_group == cta_group_2)
  {
    asm volatile(
      "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster::32b.b64 [%0], %1;"
      :
      : "r"(__as_ptr_dsmem(__smem_bar)), "r"(__ctaMask)
      : "memory");
  }
}
#endif // __cccl_ptx_isa >= 940

/*
// tcgen05.commit.cta_group.mbarrier::arrive::one.sync_restrict::shared::read::mma::a.shared::cluster.b64 [smem_bar]; //
PTX ISA 94, SM_107a, SM_107f
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_commit_sync_restrict_shared_read_mma_a(
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_commit_sync_restrict_shared_read_mma_a(
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group, ::cuda::std::uint64_t* __smem_bar)
{
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
  if constexpr (__cta_group == cta_group_1)
  {
    asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.sync_restrict::shared::read::mma::a.shared::cluster.b64 [%0];"
      :
      : "r"(__as_ptr_dsmem(__smem_bar))
      : "memory");
  }
  else if constexpr (__cta_group == cta_group_2)
  {
    asm volatile(
      "tcgen05.commit.cta_group::2.mbarrier::arrive::one.sync_restrict::shared::read::mma::a.shared::cluster.b64 [%0];"
      :
      : "r"(__as_ptr_dsmem(__smem_bar))
      : "memory");
  }
}
#endif // __cccl_ptx_isa >= 940

/*
//
tcgen05.commit.cta_group.mbarrier::arrive::one.sync_restrict::shared::read::mma::a.shared::cluster.multicast::cluster::32b.b64
[smem_bar], ctaMask; // PTX ISA 94, SM_107a, SM_107f
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_commit_sync_restrict_shared_read_mma_a_multicast_32b(
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint64_t* smem_bar,
  uint32_t ctaMask);
*/
#if __cccl_ptx_isa >= 940
template <::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_commit_sync_restrict_shared_read_mma_a_multicast_32b(
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group, ::cuda::std::uint64_t* __smem_bar, ::cuda::std::uint32_t __ctaMask)
{
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
  if constexpr (__cta_group == cta_group_1)
  {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.sync_restrict::shared::read::mma::a.shared::"
                 "cluster.multicast::cluster::32b.b64 [%0], %1;"
                 :
                 : "r"(__as_ptr_dsmem(__smem_bar)), "r"(__ctaMask)
                 : "memory");
  }
  else if constexpr (__cta_group == cta_group_2)
  {
    asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.sync_restrict::shared::read::mma::a.shared::"
                 "cluster.multicast::cluster::32b.b64 [%0], %1;"
                 :
                 : "r"(__as_ptr_dsmem(__smem_bar)), "r"(__ctaMask)
                 : "memory");
  }
}
#endif // __cccl_ptx_isa >= 940

#endif // _CUDA_PTX_GENERATED_TCGEN05_COMMIT_H_
