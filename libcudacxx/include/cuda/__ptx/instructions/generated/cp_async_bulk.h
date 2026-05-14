// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_CP_ASYNC_BULK_H_
#define _CUDA_PTX_GENERATED_CP_ASYNC_BULK_H_

/*
// cp.async.bulk.dst.src.mbarrier::complete_tx::bytes [dstMem], [srcMem], size, [smem_bar]; // PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .global }
template <typename = void>
__device__ static inline void cp_async_bulk(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* srcMem,
  const uint32_t& size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 800
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk(
  ::cuda::ptx::space_cluster_t,
  ::cuda::ptx::space_global_t,
  void* __dstMem,
  const void* __srcMem,
  const ::cuda::std::uint32_t& __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_cluster (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  asm("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
      :
      : "r"(__as_ptr_smem(__dstMem)), "l"(__as_ptr_gmem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.dst.src.mbarrier::complete_tx::bytes [dstMem], [srcMem], size, [smem_bar]; // PTX ISA 86, SM_90
// .dst       = { .shared::cta }
// .src       = { .global }
template <typename = void>
__device__ static inline void cp_async_bulk(
  cuda::ptx::space_shared_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* srcMem,
  const uint32_t& size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 860
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::space_global_t,
  void* __dstMem,
  const void* __srcMem,
  const ::cuda::std::uint32_t& __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  asm("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
      :
      : "r"(__as_ptr_smem(__dstMem)), "l"(__as_ptr_gmem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// cp.async.bulk.dst.src.mbarrier::complete_tx::bytes [dstMem], [srcMem], size, [rdsmem_bar]; // PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .shared::cta }
template <typename = void>
__device__ static inline void cp_async_bulk(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_shared_t,
  void* dstMem,
  const void* srcMem,
  const uint32_t& size,
  uint64_t* rdsmem_bar);
*/
#if __cccl_ptx_isa >= 800
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk(
  ::cuda::ptx::space_cluster_t,
  ::cuda::ptx::space_shared_t,
  void* __dstMem,
  const void* __srcMem,
  const ::cuda::std::uint32_t& __size,
  ::cuda::std::uint64_t* __rdsmem_bar)
{
  // __space == space_cluster (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  asm("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
      :
      : "r"(__as_ptr_remote_dsmem(__dstMem)),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_remote_dsmem(__rdsmem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.dst.src.mbarrier::complete_tx::bytes.ignore_oob [dstMem], [srcMem], size, ignoreBytesLeft,
ignoreBytesRight, [smem_bar]; // PTX ISA 92, SM_90
// .dst       = { .shared::cta }
// .src       = { .global }
template <typename = void>
__device__ static inline void cp_async_bulk_ignore_oob(
  cuda::ptx::space_shared_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* srcMem,
  const uint32_t& size,
  const uint32_t& ignoreBytesLeft,
  const uint32_t& ignoreBytesRight,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 920
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_ignore_oob(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::space_global_t,
  void* __dstMem,
  const void* __srcMem,
  const ::cuda::std::uint32_t& __size,
  const ::cuda::std::uint32_t& __ignoreBytesLeft,
  const ::cuda::std::uint32_t& __ignoreBytesRight,
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  asm("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.ignore_oob [%0], [%1], %2, %3, %4, [%5];"
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__as_ptr_gmem(__srcMem)),
        "r"(__size),
        "r"(__ignoreBytesLeft),
        "r"(__ignoreBytesRight),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 920

/*
// cp.async.bulk.dst.src.bulk_group [dstMem], [srcMem], size; // PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
template <typename = void>
__device__ static inline void cp_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  void* dstMem,
  const void* srcMem,
  const uint32_t& size);
*/
#if __cccl_ptx_isa >= 800
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk(
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  void* __dstMem,
  const void* __srcMem,
  const ::cuda::std::uint32_t& __size)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  asm("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
      :
      : "l"(__as_ptr_gmem(__dstMem)), "r"(__as_ptr_smem(__srcMem)), "r"(__size)
      : "memory");
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.dst.src.bulk_group.cp_mask [dstMem], [srcMem], size, byteMask; // PTX ISA 86, SM_100
// .dst       = { .global }
// .src       = { .shared::cta }
template <typename = void>
__device__ static inline void cp_async_bulk_cp_mask(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  void* dstMem,
  const void* srcMem,
  const uint32_t& size,
  const uint16_t& byteMask);
*/
#if __cccl_ptx_isa >= 860
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_cp_mask(
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  void* __dstMem,
  const void* __srcMem,
  const ::cuda::std::uint32_t& __size,
  const ::cuda::std::uint16_t& __byteMask)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  asm("cp.async.bulk.global.shared::cta.bulk_group.cp_mask [%0], [%1], %2, %3;"
      :
      : "l"(__as_ptr_gmem(__dstMem)), "r"(__as_ptr_smem(__srcMem)), "r"(__size), "h"(__byteMask)
      : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// cp.async.bulk.dst.src.mbarrier::complete_tx::bytes.report_mechanism [dstMem], [srcMem], size, [smem_bar]; // PTX ISA
94, SM_107a, SM_107f
// .dst       = { .shared::cta, .shared::cluster }
// .src       = { .global }
// .report_mechanism = { .mbarrier::report::validity::per_16bytes::80000000,
.mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80,
.mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff } template
<cuda::ptx::dot_space Space, cuda::ptx::dot_report_mechanism Report_Mechanism>
__device__ static inline void cp_async_bulk(
  cuda::ptx::space_t<Space> space,
  cuda::ptx::space_global_t,
  cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
  void* dstMem,
  const void* srcMem,
  const uint32_t& size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <::cuda::ptx::dot_space _Space, ::cuda::ptx::dot_report_mechanism _Report_Mechanism>
_CCCL_DEVICE static inline void cp_async_bulk(
  ::cuda::ptx::space_t<_Space> __space,
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::report_mechanism_t<_Report_Mechanism> __report_mechanism,
  void* __dstMem,
  const void* __srcMem,
  const ::cuda::std::uint32_t& __size,
  ::cuda::std::uint64_t* __smem_bar)
{
  static_assert(__space == space_shared || __space == space_cluster, "");
  // __space == space_global (due to parameter type constraint)
  static_assert(
    __report_mechanism == mbarrier_report_valid_per_16bytes_80000000
      || __report_mechanism == mbarrier_report_valid_per_16bytes_8000
      || __report_mechanism == mbarrier_report_valid_per_16bytes_80
      || __report_mechanism == mbarrier_report_valid_per_16bytes_8
      || __report_mechanism == mbarrier_report_valid_per_element_ff,
    "");
  if constexpr (__space == space_shared && __report_mechanism == mbarrier_report_valid_per_16bytes_80000000)
  {
    asm("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.mbarrier::report::validity::per_16bytes::"
        "80000000 [%0], [%1], %2, [%3];"
        :
        : "r"(__as_ptr_smem(__dstMem)), "l"(__as_ptr_gmem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
        : "memory");
  }
  else if constexpr (__space == space_shared && __report_mechanism == mbarrier_report_valid_per_16bytes_8000)
  {
    asm("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.mbarrier::report::validity::per_16bytes::8000 "
        "[%0], [%1], %2, [%3];"
        :
        : "r"(__as_ptr_smem(__dstMem)), "l"(__as_ptr_gmem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
        : "memory");
  }
  else if constexpr (__space == space_shared && __report_mechanism == mbarrier_report_valid_per_16bytes_80)
  {
    asm("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.mbarrier::report::validity::per_16bytes::80 "
        "[%0], [%1], %2, [%3];"
        :
        : "r"(__as_ptr_smem(__dstMem)), "l"(__as_ptr_gmem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
        : "memory");
  }
  else if constexpr (__space == space_shared && __report_mechanism == mbarrier_report_valid_per_16bytes_8)
  {
    asm("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.mbarrier::report::validity::per_16bytes::8 "
        "[%0], [%1], %2, [%3];"
        :
        : "r"(__as_ptr_smem(__dstMem)), "l"(__as_ptr_gmem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
        : "memory");
  }
  else if constexpr (__space == space_shared && __report_mechanism == mbarrier_report_valid_per_element_ff)
  {
    asm("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.mbarrier::report::validity::per_element::ff "
        "[%0], [%1], %2, [%3];"
        :
        : "r"(__as_ptr_smem(__dstMem)), "l"(__as_ptr_gmem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
        : "memory");
  }
  else if constexpr (__space == space_cluster && __report_mechanism == mbarrier_report_valid_per_16bytes_80000000)
  {
    asm("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.mbarrier::report::validity::per_16bytes::"
        "80000000 [%0], [%1], %2, [%3];"
        :
        : "r"(__as_ptr_smem(__dstMem)), "l"(__as_ptr_gmem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
        : "memory");
  }
  else if constexpr (__space == space_cluster && __report_mechanism == mbarrier_report_valid_per_16bytes_8000)
  {
    asm("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.mbarrier::report::validity::per_16bytes::"
        "8000 [%0], [%1], %2, [%3];"
        :
        : "r"(__as_ptr_smem(__dstMem)), "l"(__as_ptr_gmem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
        : "memory");
  }
  else if constexpr (__space == space_cluster && __report_mechanism == mbarrier_report_valid_per_16bytes_80)
  {
    asm("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.mbarrier::report::validity::per_16bytes::80 "
        "[%0], [%1], %2, [%3];"
        :
        : "r"(__as_ptr_smem(__dstMem)), "l"(__as_ptr_gmem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
        : "memory");
  }
  else if constexpr (__space == space_cluster && __report_mechanism == mbarrier_report_valid_per_16bytes_8)
  {
    asm("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.mbarrier::report::validity::per_16bytes::8 "
        "[%0], [%1], %2, [%3];"
        :
        : "r"(__as_ptr_smem(__dstMem)), "l"(__as_ptr_gmem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
        : "memory");
  }
  else if constexpr (__space == space_cluster && __report_mechanism == mbarrier_report_valid_per_element_ff)
  {
    asm("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.mbarrier::report::validity::per_element::ff "
        "[%0], [%1], %2, [%3];"
        :
        : "r"(__as_ptr_smem(__dstMem)), "l"(__as_ptr_gmem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
        : "memory");
  }
}
#endif // __cccl_ptx_isa >= 940

#endif // _CUDA_PTX_GENERATED_CP_ASYNC_BULK_H_
