// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_CP_ASYNC_BULK_TENSOR_GATHER_SCATTER_H_
#define _CUDA_PTX_GENERATED_CP_ASYNC_BULK_TENSOR_GATHER_SCATTER_H_

/*
// cp.async.bulk.tensor.2d.dst.src.tile::gather4.mbarrier::complete_tx::bytes [dstMem], [tensorMap, tensorCoords],
[smem_bar]; // PTX ISA 86, SM_100
// .dst       = { .shared::cta }
// .src       = { .global }
template <typename = void>
__device__ static inline void cp_async_bulk_tensor_tile_gather4(
  cuda::ptx::space_shared_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[5],
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 860
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor_tile_gather4(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::space_global_t,
  void* __dstMem,
  const void* __tensorMap,
  const ::cuda::std::int32_t (&__tensorCoords)[5],
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  asm("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, "
      "%5, %6}], [%7];"
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__tensorCoords[4]),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// cp.async.bulk.tensor.2d.dst.src.tile::gather4.mbarrier::complete_tx::bytes.cta_group [dstMem], [tensorMap,
tensorCoords], [smem_bar]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
// .dst       = { .shared::cta }
// .src       = { .global }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void cp_async_bulk_tensor_tile_gather4(
  cuda::ptx::space_shared_t,
  cuda::ptx::space_global_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[5],
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 860
template <::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void cp_async_bulk_tensor_tile_gather4(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group,
  void* __dstMem,
  const void* __tensorMap,
  const ::cuda::std::int32_t (&__tensorCoords)[5],
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
  if constexpr (__cta_group == cta_group_1)
  {
    asm("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::1 [%0], [%1, "
        "{%2, %3, %4, %5, %6}], [%7];"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar))
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2)
  {
    asm("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::2 [%0], [%1, "
        "{%2, %3, %4, %5, %6}], [%7];"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar))
        : "memory");
  }
}
#endif // __cccl_ptx_isa >= 860

/*
// cp.async.bulk.tensor.2d.dst.src.tile::gather4.mbarrier::complete_tx::bytes.multicast::cluster [dstMem], [tensorMap,
tensorCoords], [smem_bar], ctaMask; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a,
SM_110f
// .dst       = { .shared::cluster }
// .src       = { .global }
template <typename = void>
__device__ static inline void cp_async_bulk_tensor_tile_gather4(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[5],
  uint64_t* smem_bar,
  const uint16_t& ctaMask);
*/
#if __cccl_ptx_isa >= 860
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor_tile_gather4(
  ::cuda::ptx::space_cluster_t,
  ::cuda::ptx::space_global_t,
  void* __dstMem,
  const void* __tensorMap,
  const ::cuda::std::int32_t (&__tensorCoords)[5],
  ::cuda::std::uint64_t* __smem_bar,
  const ::cuda::std::uint16_t& __ctaMask)
{
  // __space == space_cluster (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes.multicast::cluster "
      "[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__tensorCoords[4]),
        "r"(__as_ptr_smem(__smem_bar)),
        "h"(__ctaMask)
      : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// cp.async.bulk.tensor.2d.dst.src.tile::gather4.mbarrier::complete_tx::bytes.multicast::cluster.cta_group [dstMem],
[tensorMap, tensorCoords], [smem_bar], ctaMask; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f,
SM_110a, SM_110f
// .dst       = { .shared::cluster }
// .src       = { .global }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void cp_async_bulk_tensor_tile_gather4(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[5],
  uint64_t* smem_bar,
  const uint16_t& ctaMask);
*/
#if __cccl_ptx_isa >= 860
template <::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void cp_async_bulk_tensor_tile_gather4(
  ::cuda::ptx::space_cluster_t,
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group,
  void* __dstMem,
  const void* __tensorMap,
  const ::cuda::std::int32_t (&__tensorCoords)[5],
  ::cuda::std::uint64_t* __smem_bar,
  const ::cuda::std::uint16_t& __ctaMask)
{
  // __space == space_cluster (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
  if constexpr (__cta_group == cta_group_1)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes.multicast::cluster."
        "cta_group::1 [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar)),
          "h"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes.multicast::cluster."
        "cta_group::2 [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar)),
          "h"(__ctaMask)
        : "memory");
  }
}
#endif // __cccl_ptx_isa >= 860

/*
// cp.async.bulk.tensor.2d.dst.src.tile::scatter4.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 86,
SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
// .dst       = { .global }
// .src       = { .shared::cta }
template <typename = void>
__device__ static inline void cp_async_bulk_tensor_tile_scatter4(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  const void* tensorMap,
  const int32_t (&tensorCoords)[5],
  const void* srcMem);
*/
#if __cccl_ptx_isa >= 860
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor_tile_scatter4(
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  const void* __tensorMap,
  const ::cuda::std::int32_t (&__tensorCoords)[5],
  const void* __srcMem)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  asm("cp.async.bulk.tensor.2d.global.shared::cta.tile::scatter4.bulk_group [%0, {%1, %2, %3, %4, %5}], [%6];"
      :
      : "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__tensorCoords[4]),
        "r"(__as_ptr_smem(__srcMem))
      : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// cp.async.bulk.tensor.2d.dst.src.tile::gather4.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem],
[tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
// .dst       = { .shared::cta }
// .src       = { .global }
// .cta_group = { .cta_group::1, .cta_group::2 }
// .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000,
.mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80,
.mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff } template
<cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
__device__ static inline void cp_async_bulk_tensor_tile_gather4(
  cuda::ptx::space_shared_t,
  cuda::ptx::space_global_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[5],
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <::cuda::ptx::dot_cta_group _Cta_Group, ::cuda::ptx::dot_report_mechanism _Report_Mechanism>
_CCCL_DEVICE static inline void cp_async_bulk_tensor_tile_gather4(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group,
  ::cuda::ptx::report_mechanism_t<_Report_Mechanism> __report_mechanism,
  void* __dstMem,
  const void* __tensorMap,
  const ::cuda::std::int32_t (&__tensorCoords)[5],
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
  static_assert(
    __report_mechanism == mbarrier_report_disabled || __report_mechanism == mbarrier_report_valid_per_16bytes_80000000
      || __report_mechanism == mbarrier_report_valid_per_16bytes_8000
      || __report_mechanism == mbarrier_report_valid_per_16bytes_80
      || __report_mechanism == mbarrier_report_valid_per_16bytes_8
      || __report_mechanism == mbarrier_report_valid_per_element_ff,
    "");
  if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_disabled)
  {
    asm("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::"
        "report::disabled [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar))
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_80000000)
  {
    asm("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::"
        "report::validity::per_16bytes::80000000 [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar))
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_8000)
  {
    asm("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::"
        "report::validity::per_16bytes::8000 [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar))
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_80)
  {
    asm("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::"
        "report::validity::per_16bytes::80 [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar))
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_8)
  {
    asm("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::"
        "report::validity::per_16bytes::8 [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar))
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_element_ff)
  {
    asm("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::"
        "report::validity::per_element::ff [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar))
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_disabled)
  {
    asm("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::"
        "report::disabled [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar))
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_80000000)
  {
    asm("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::"
        "report::validity::per_16bytes::80000000 [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar))
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_8000)
  {
    asm("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::"
        "report::validity::per_16bytes::8000 [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar))
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_80)
  {
    asm("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::"
        "report::validity::per_16bytes::80 [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar))
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_8)
  {
    asm("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::"
        "report::validity::per_16bytes::8 [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar))
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_element_ff)
  {
    asm("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::"
        "report::validity::per_element::ff [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar))
        : "memory");
  }
}
#endif // __cccl_ptx_isa >= 940

/*
//
cp.async.bulk.tensor.2d.dst.src.tile::gather4.mbarrier::complete_tx::bytes.multicast::cluster::32b.cta_group.report_mechanism
[dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask; // PTX ISA 94, SM_107a, SM_107f
// .dst       = { .shared::cluster }
// .src       = { .global }
// .cta_group = { .cta_group::1, .cta_group::2 }
// .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000,
.mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80,
.mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff } template
<cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
__device__ static inline void cp_async_bulk_tensor_tile_gather4_multicast_32b(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[5],
  uint64_t* smem_bar,
  const uint32_t& ctaMask);
*/
#if __cccl_ptx_isa >= 940
template <::cuda::ptx::dot_cta_group _Cta_Group, ::cuda::ptx::dot_report_mechanism _Report_Mechanism>
_CCCL_DEVICE static inline void cp_async_bulk_tensor_tile_gather4_multicast_32b(
  ::cuda::ptx::space_cluster_t,
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group,
  ::cuda::ptx::report_mechanism_t<_Report_Mechanism> __report_mechanism,
  void* __dstMem,
  const void* __tensorMap,
  const ::cuda::std::int32_t (&__tensorCoords)[5],
  ::cuda::std::uint64_t* __smem_bar,
  const ::cuda::std::uint32_t& __ctaMask)
{
  // __space == space_cluster (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
  static_assert(
    __report_mechanism == mbarrier_report_disabled || __report_mechanism == mbarrier_report_valid_per_16bytes_80000000
      || __report_mechanism == mbarrier_report_valid_per_16bytes_8000
      || __report_mechanism == mbarrier_report_valid_per_16bytes_80
      || __report_mechanism == mbarrier_report_valid_per_16bytes_8
      || __report_mechanism == mbarrier_report_valid_per_element_ff,
    "");
  if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_disabled)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes.multicast::cluster::"
        "32b.cta_group::1.mbarrier::report::disabled [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_80000000)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes.multicast::cluster::"
        "32b.cta_group::1.mbarrier::report::validity::per_16bytes::80000000 [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_8000)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes.multicast::cluster::"
        "32b.cta_group::1.mbarrier::report::validity::per_16bytes::8000 [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_80)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes.multicast::cluster::"
        "32b.cta_group::1.mbarrier::report::validity::per_16bytes::80 [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_8)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes.multicast::cluster::"
        "32b.cta_group::1.mbarrier::report::validity::per_16bytes::8 [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_element_ff)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes.multicast::cluster::"
        "32b.cta_group::1.mbarrier::report::validity::per_element::ff [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_disabled)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes.multicast::cluster::"
        "32b.cta_group::2.mbarrier::report::disabled [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_80000000)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes.multicast::cluster::"
        "32b.cta_group::2.mbarrier::report::validity::per_16bytes::80000000 [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_8000)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes.multicast::cluster::"
        "32b.cta_group::2.mbarrier::report::validity::per_16bytes::8000 [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_80)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes.multicast::cluster::"
        "32b.cta_group::2.mbarrier::report::validity::per_16bytes::80 [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_8)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes.multicast::cluster::"
        "32b.cta_group::2.mbarrier::report::validity::per_16bytes::8 [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_element_ff)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes.multicast::cluster::"
        "32b.cta_group::2.mbarrier::report::validity::per_element::ff [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
}
#endif // __cccl_ptx_isa >= 940

/*
//
cp.async.bulk.tensor.2d.dst.src.tile::gather4.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address
[dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
// .dst       = { .shared::cta }
// .src       = { .global }
// .cta_group = { .cta_group::1, .cta_group::2 }
// .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000,
.mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80,
.mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff } template
<cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
__device__ static inline void cp_async_bulk_tensor_tile_gather4_override(
  cuda::ptx::space_shared_t,
  cuda::ptx::space_global_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
  void* dstMem,
  const void* tensorMap,
  const void* gAddrToOverride,
  const int32_t (&tensorCoords)[5],
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 940
template <::cuda::ptx::dot_cta_group _Cta_Group, ::cuda::ptx::dot_report_mechanism _Report_Mechanism>
_CCCL_DEVICE static inline void cp_async_bulk_tensor_tile_gather4_override(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group,
  ::cuda::ptx::report_mechanism_t<_Report_Mechanism> __report_mechanism,
  void* __dstMem,
  const void* __tensorMap,
  const void* __gAddrToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[5],
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
  static_assert(
    __report_mechanism == mbarrier_report_disabled || __report_mechanism == mbarrier_report_valid_per_16bytes_80000000
      || __report_mechanism == mbarrier_report_valid_per_16bytes_8000
      || __report_mechanism == mbarrier_report_valid_per_16bytes_80
      || __report_mechanism == mbarrier_report_valid_per_16bytes_8
      || __report_mechanism == mbarrier_report_valid_per_element_ff,
    "");
  if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_disabled)
  {
    asm("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::"
        "report::disabled.override::global_address [%0], [%1, %2, {%3, %4, %5, %6, %7}], [%8];"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar))
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_80000000)
  {
    asm("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::"
        "report::validity::per_16bytes::80000000.override::global_address [%0], [%1, %2, {%3, %4, %5, %6, %7}], [%8];"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar))
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_8000)
  {
    asm("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::"
        "report::validity::per_16bytes::8000.override::global_address [%0], [%1, %2, {%3, %4, %5, %6, %7}], [%8];"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar))
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_80)
  {
    asm("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::"
        "report::validity::per_16bytes::80.override::global_address [%0], [%1, %2, {%3, %4, %5, %6, %7}], [%8];"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar))
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_8)
  {
    asm("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::"
        "report::validity::per_16bytes::8.override::global_address [%0], [%1, %2, {%3, %4, %5, %6, %7}], [%8];"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar))
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_element_ff)
  {
    asm("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::"
        "report::validity::per_element::ff.override::global_address [%0], [%1, %2, {%3, %4, %5, %6, %7}], [%8];"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar))
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_disabled)
  {
    asm("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::"
        "report::disabled.override::global_address [%0], [%1, %2, {%3, %4, %5, %6, %7}], [%8];"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar))
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_80000000)
  {
    asm("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::"
        "report::validity::per_16bytes::80000000.override::global_address [%0], [%1, %2, {%3, %4, %5, %6, %7}], [%8];"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar))
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_8000)
  {
    asm("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::"
        "report::validity::per_16bytes::8000.override::global_address [%0], [%1, %2, {%3, %4, %5, %6, %7}], [%8];"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar))
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_80)
  {
    asm("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::"
        "report::validity::per_16bytes::80.override::global_address [%0], [%1, %2, {%3, %4, %5, %6, %7}], [%8];"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar))
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_8)
  {
    asm("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::"
        "report::validity::per_16bytes::8.override::global_address [%0], [%1, %2, {%3, %4, %5, %6, %7}], [%8];"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar))
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_element_ff)
  {
    asm("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::"
        "report::validity::per_element::ff.override::global_address [%0], [%1, %2, {%3, %4, %5, %6, %7}], [%8];"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar))
        : "memory");
  }
}
#endif // __cccl_ptx_isa >= 940

/*
//
cp.async.bulk.tensor.2d.dst.src.tile::gather4.mbarrier::complete_tx::bytes.multicast::cluster::32b.cta_group.report_mechanism.override::global_address
[dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar], ctaMask; // PTX ISA 94, SM_107a, SM_107f
// .dst       = { .shared::cluster }
// .src       = { .global }
// .cta_group = { .cta_group::1, .cta_group::2 }
// .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000,
.mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80,
.mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff } template
<cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
__device__ static inline void cp_async_bulk_tensor_tile_gather4_multicast_32b_override(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
  void* dstMem,
  const void* tensorMap,
  const void* gAddrToOverride,
  const int32_t (&tensorCoords)[5],
  uint64_t* smem_bar,
  const uint32_t& ctaMask);
*/
#if __cccl_ptx_isa >= 940
template <::cuda::ptx::dot_cta_group _Cta_Group, ::cuda::ptx::dot_report_mechanism _Report_Mechanism>
_CCCL_DEVICE static inline void cp_async_bulk_tensor_tile_gather4_multicast_32b_override(
  ::cuda::ptx::space_cluster_t,
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group,
  ::cuda::ptx::report_mechanism_t<_Report_Mechanism> __report_mechanism,
  void* __dstMem,
  const void* __tensorMap,
  const void* __gAddrToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[5],
  ::cuda::std::uint64_t* __smem_bar,
  const ::cuda::std::uint32_t& __ctaMask)
{
  // __space == space_cluster (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
  static_assert(
    __report_mechanism == mbarrier_report_disabled || __report_mechanism == mbarrier_report_valid_per_16bytes_80000000
      || __report_mechanism == mbarrier_report_valid_per_16bytes_8000
      || __report_mechanism == mbarrier_report_valid_per_16bytes_80
      || __report_mechanism == mbarrier_report_valid_per_16bytes_8
      || __report_mechanism == mbarrier_report_valid_per_element_ff,
    "");
  if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_disabled)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes.multicast::cluster::"
        "32b.cta_group::1.mbarrier::report::disabled.override::global_address [%0], [%1, %2, {%3, %4, %5, %6, %7}], "
        "[%8], %9;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_80000000)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes.multicast::cluster::"
        "32b.cta_group::1.mbarrier::report::validity::per_16bytes::80000000.override::global_address [%0], [%1, %2, "
        "{%3, %4, %5, %6, %7}], [%8], %9;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_8000)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes.multicast::cluster::"
        "32b.cta_group::1.mbarrier::report::validity::per_16bytes::8000.override::global_address [%0], [%1, %2, {%3, "
        "%4, %5, %6, %7}], [%8], %9;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_80)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes.multicast::cluster::"
        "32b.cta_group::1.mbarrier::report::validity::per_16bytes::80.override::global_address [%0], [%1, %2, {%3, %4, "
        "%5, %6, %7}], [%8], %9;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_8)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes.multicast::cluster::"
        "32b.cta_group::1.mbarrier::report::validity::per_16bytes::8.override::global_address [%0], [%1, %2, {%3, %4, "
        "%5, %6, %7}], [%8], %9;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_element_ff)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes.multicast::cluster::"
        "32b.cta_group::1.mbarrier::report::validity::per_element::ff.override::global_address [%0], [%1, %2, {%3, %4, "
        "%5, %6, %7}], [%8], %9;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_disabled)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes.multicast::cluster::"
        "32b.cta_group::2.mbarrier::report::disabled.override::global_address [%0], [%1, %2, {%3, %4, %5, %6, %7}], "
        "[%8], %9;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_80000000)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes.multicast::cluster::"
        "32b.cta_group::2.mbarrier::report::validity::per_16bytes::80000000.override::global_address [%0], [%1, %2, "
        "{%3, %4, %5, %6, %7}], [%8], %9;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_8000)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes.multicast::cluster::"
        "32b.cta_group::2.mbarrier::report::validity::per_16bytes::8000.override::global_address [%0], [%1, %2, {%3, "
        "%4, %5, %6, %7}], [%8], %9;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_80)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes.multicast::cluster::"
        "32b.cta_group::2.mbarrier::report::validity::per_16bytes::80.override::global_address [%0], [%1, %2, {%3, %4, "
        "%5, %6, %7}], [%8], %9;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_8)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes.multicast::cluster::"
        "32b.cta_group::2.mbarrier::report::validity::per_16bytes::8.override::global_address [%0], [%1, %2, {%3, %4, "
        "%5, %6, %7}], [%8], %9;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_element_ff)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes.multicast::cluster::"
        "32b.cta_group::2.mbarrier::report::validity::per_element::ff.override::global_address [%0], [%1, %2, {%3, %4, "
        "%5, %6, %7}], [%8], %9;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
}
#endif // __cccl_ptx_isa >= 940

/*
// cp.async.bulk.tensor.2d.dst.src.tile::scatter4.bulk_group.override::global_address [tensorMap, gAddrToOverride,
tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
// .dst       = { .global }
// .src       = { .shared::cta }
template <typename = void>
__device__ static inline void cp_async_bulk_tensor_tile_scatter4_override(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  const void* tensorMap,
  const void* gAddrToOverride,
  const int32_t (&tensorCoords)[5],
  const void* srcMem);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor_tile_scatter4_override(
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  const void* __tensorMap,
  const void* __gAddrToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[5],
  const void* __srcMem)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  asm("cp.async.bulk.tensor.2d.global.shared::cta.tile::scatter4.bulk_group.override::global_address [%0, %1, {%2, %3, "
      "%4, %5, %6}], [%7];"
      :
      : "l"(__tensorMap),
        "l"(__gAddrToOverride),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__tensorCoords[4]),
        "r"(__as_ptr_smem(__srcMem))
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

#endif // _CUDA_PTX_GENERATED_CP_ASYNC_BULK_TENSOR_GATHER_SCATTER_H_
