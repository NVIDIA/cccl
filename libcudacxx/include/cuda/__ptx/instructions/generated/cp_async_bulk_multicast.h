// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_CP_ASYNC_BULK_MULTICAST_H_
#define _CUDA_PTX_GENERATED_CP_ASYNC_BULK_MULTICAST_H_

/*
// cp.async.bulk.dst.src.mbarrier::complete_tx::bytes.multicast::cluster [dstMem], [srcMem], size, [smem_bar], ctaMask;
// PTX ISA 80, SM_90a, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
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
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk(
  ::cuda::ptx::space_cluster_t,
  ::cuda::ptx::space_global_t,
  void* __dstMem,
  const void* __srcMem,
  const ::cuda::std::uint32_t& __size,
  ::cuda::std::uint64_t* __smem_bar,
  const ::cuda::std::uint16_t& __ctaMask)
{
  // __space == space_cluster (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  asm("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1], %2, [%3], %4;"
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__as_ptr_gmem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar)),
        "h"(__ctaMask)
      : "memory");
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.dst.src.mbarrier::complete_tx::bytes.multicast::cluster::32b [dstMem], [srcMem], size, [smem_bar],
ctaMask; // PTX ISA 94, SM_107a, SM_107f
// .dst       = { .shared::cluster }
// .src       = { .global }
template <typename = void>
__device__ static inline void cp_async_bulk_multicast_32b(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* srcMem,
  const uint32_t& size,
  uint64_t* smem_bar,
  const uint32_t& ctaMask);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_multicast_32b(
  ::cuda::ptx::space_cluster_t,
  ::cuda::ptx::space_global_t,
  void* __dstMem,
  const void* __srcMem,
  const ::cuda::std::uint32_t& __size,
  ::cuda::std::uint64_t* __smem_bar,
  const ::cuda::std::uint32_t& __ctaMask)
{
  // __space == space_cluster (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  asm("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster::32b [%0], [%1], %2, [%3], "
      "%4;"
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__as_ptr_gmem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar)),
        "r"(__ctaMask)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// cp.async.bulk.dst.src.mbarrier::complete_tx::bytes.multicast::cluster::32b.report_mechanism [dstMem], [srcMem], size,
[smem_bar], ctaMask; // PTX ISA 94, SM_107a, SM_107f
// .dst       = { .shared::cluster }
// .src       = { .global }
// .report_mechanism = { .mbarrier::report::validity::per_16bytes::80000000,
.mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80,
.mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff } template
<cuda::ptx::dot_report_mechanism Report_Mechanism>
__device__ static inline void cp_async_bulk_multicast_32b(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
  void* dstMem,
  const void* srcMem,
  const uint32_t& size,
  uint64_t* smem_bar,
  const uint32_t& ctaMask);
*/
#if __cccl_ptx_isa >= 940
template <::cuda::ptx::dot_report_mechanism _Report_Mechanism>
_CCCL_DEVICE static inline void cp_async_bulk_multicast_32b(
  ::cuda::ptx::space_cluster_t,
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::report_mechanism_t<_Report_Mechanism> __report_mechanism,
  void* __dstMem,
  const void* __srcMem,
  const ::cuda::std::uint32_t& __size,
  ::cuda::std::uint64_t* __smem_bar,
  const ::cuda::std::uint32_t& __ctaMask)
{
  // __space == space_cluster (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  static_assert(
    __report_mechanism == mbarrier_report_valid_per_16bytes_80000000
      || __report_mechanism == mbarrier_report_valid_per_16bytes_8000
      || __report_mechanism == mbarrier_report_valid_per_16bytes_80
      || __report_mechanism == mbarrier_report_valid_per_16bytes_8
      || __report_mechanism == mbarrier_report_valid_per_element_ff,
    "");
  if constexpr (__report_mechanism == mbarrier_report_valid_per_16bytes_80000000)
  {
    asm("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster::32b.mbarrier::report::"
        "validity::per_16bytes::80000000 [%0], [%1], %2, [%3], %4;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__as_ptr_gmem(__srcMem)),
          "r"(__size),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__report_mechanism == mbarrier_report_valid_per_16bytes_8000)
  {
    asm("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster::32b.mbarrier::report::"
        "validity::per_16bytes::8000 [%0], [%1], %2, [%3], %4;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__as_ptr_gmem(__srcMem)),
          "r"(__size),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__report_mechanism == mbarrier_report_valid_per_16bytes_80)
  {
    asm("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster::32b.mbarrier::report::"
        "validity::per_16bytes::80 [%0], [%1], %2, [%3], %4;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__as_ptr_gmem(__srcMem)),
          "r"(__size),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__report_mechanism == mbarrier_report_valid_per_16bytes_8)
  {
    asm("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster::32b.mbarrier::report::"
        "validity::per_16bytes::8 [%0], [%1], %2, [%3], %4;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__as_ptr_gmem(__srcMem)),
          "r"(__size),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__report_mechanism == mbarrier_report_valid_per_element_ff)
  {
    asm("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster::32b.mbarrier::report::"
        "validity::per_element::ff [%0], [%1], %2, [%3], %4;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__as_ptr_gmem(__srcMem)),
          "r"(__size),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
}
#endif // __cccl_ptx_isa >= 940

#endif // _CUDA_PTX_GENERATED_CP_ASYNC_BULK_MULTICAST_H_
