// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_CP_ASYNC_BULK_TENSOR_MULTICAST_H_
#define _CUDA_PTX_GENERATED_CP_ASYNC_BULK_TENSOR_MULTICAST_H_

/*
// cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem], [tensorMap,
tensorCoords], [smem_bar], ctaMask; // PTX ISA 80, SM_90a, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f,
SM_110a, SM_110f
// .dst       = { .shared::cluster }
// .src       = { .global }
template <typename = void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[1],
  uint64_t* smem_bar,
  const uint16_t& ctaMask);
*/
#if __cccl_ptx_isa >= 800
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  ::cuda::ptx::space_cluster_t,
  ::cuda::ptx::space_global_t,
  void* __dstMem,
  const void* __tensorMap,
  const ::cuda::std::int32_t (&__tensorCoords)[1],
  ::cuda::std::uint64_t* __smem_bar,
  const ::cuda::std::uint16_t& __ctaMask)
{
  // __space == space_cluster (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  asm("cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, "
      "{%2}], [%3], %4;"
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__as_ptr_smem(__smem_bar)),
        "h"(__ctaMask)
      : "memory");
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem], [tensorMap,
tensorCoords], [smem_bar], ctaMask; // PTX ISA 80, SM_90a, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f,
SM_110a, SM_110f
// .dst       = { .shared::cluster }
// .src       = { .global }
template <typename = void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[2],
  uint64_t* smem_bar,
  const uint16_t& ctaMask);
*/
#if __cccl_ptx_isa >= 800
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  ::cuda::ptx::space_cluster_t,
  ::cuda::ptx::space_global_t,
  void* __dstMem,
  const void* __tensorMap,
  const ::cuda::std::int32_t (&__tensorCoords)[2],
  ::cuda::std::uint64_t* __smem_bar,
  const ::cuda::std::uint16_t& __ctaMask)
{
  // __space == space_cluster (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, "
      "{%2, %3}], [%4], %5;"
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__as_ptr_smem(__smem_bar)),
        "h"(__ctaMask)
      : "memory");
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem], [tensorMap,
tensorCoords], [smem_bar], ctaMask; // PTX ISA 80, SM_90a, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f,
SM_110a, SM_110f
// .dst       = { .shared::cluster }
// .src       = { .global }
template <typename = void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[3],
  uint64_t* smem_bar,
  const uint16_t& ctaMask);
*/
#if __cccl_ptx_isa >= 800
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  ::cuda::ptx::space_cluster_t,
  ::cuda::ptx::space_global_t,
  void* __dstMem,
  const void* __tensorMap,
  const ::cuda::std::int32_t (&__tensorCoords)[3],
  ::cuda::std::uint64_t* __smem_bar,
  const ::cuda::std::uint16_t& __ctaMask)
{
  // __space == space_cluster (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  asm("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, "
      "{%2, %3, %4}], [%5], %6;"
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__as_ptr_smem(__smem_bar)),
        "h"(__ctaMask)
      : "memory");
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem], [tensorMap,
tensorCoords], [smem_bar], ctaMask; // PTX ISA 80, SM_90a, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f,
SM_110a, SM_110f
// .dst       = { .shared::cluster }
// .src       = { .global }
template <typename = void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[4],
  uint64_t* smem_bar,
  const uint16_t& ctaMask);
*/
#if __cccl_ptx_isa >= 800
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  ::cuda::ptx::space_cluster_t,
  ::cuda::ptx::space_global_t,
  void* __dstMem,
  const void* __tensorMap,
  const ::cuda::std::int32_t (&__tensorCoords)[4],
  ::cuda::std::uint64_t* __smem_bar,
  const ::cuda::std::uint16_t& __ctaMask)
{
  // __space == space_cluster (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  asm("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, "
      "{%2, %3, %4, %5}], [%6], %7;"
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__as_ptr_smem(__smem_bar)),
        "h"(__ctaMask)
      : "memory");
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem], [tensorMap,
tensorCoords], [smem_bar], ctaMask; // PTX ISA 80, SM_90a, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f,
SM_110a, SM_110f
// .dst       = { .shared::cluster }
// .src       = { .global }
template <typename = void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[5],
  uint64_t* smem_bar,
  const uint16_t& ctaMask);
*/
#if __cccl_ptx_isa >= 800
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
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
  asm("cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, "
      "{%2, %3, %4, %5, %6}], [%7], %8;"
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
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group [dstMem], [tensorMap,
tensorCoords], [smem_bar], ctaMask; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a,
SM_110f
// .dst       = { .shared::cluster }
// .src       = { .global }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[1],
  uint64_t* smem_bar,
  const uint16_t& ctaMask);
*/
#if __cccl_ptx_isa >= 860
template <::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  ::cuda::ptx::space_cluster_t,
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group,
  void* __dstMem,
  const void* __tensorMap,
  const ::cuda::std::int32_t (&__tensorCoords)[1],
  ::cuda::std::uint64_t* __smem_bar,
  const ::cuda::std::uint16_t& __ctaMask)
{
  // __space == space_cluster (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
  if constexpr (__cta_group == cta_group_1)
  {
    asm("cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group:"
        ":1 [%0], [%1, {%2}], [%3], %4;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__smem_bar)),
          "h"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2)
  {
    asm("cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group:"
        ":2 [%0], [%1, {%2}], [%3], %4;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__smem_bar)),
          "h"(__ctaMask)
        : "memory");
  }
}
#endif // __cccl_ptx_isa >= 860

/*
// cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group [dstMem], [tensorMap,
tensorCoords], [smem_bar], ctaMask; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a,
SM_110f
// .dst       = { .shared::cluster }
// .src       = { .global }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[2],
  uint64_t* smem_bar,
  const uint16_t& ctaMask);
*/
#if __cccl_ptx_isa >= 860
template <::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  ::cuda::ptx::space_cluster_t,
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group,
  void* __dstMem,
  const void* __tensorMap,
  const ::cuda::std::int32_t (&__tensorCoords)[2],
  ::cuda::std::uint64_t* __smem_bar,
  const ::cuda::std::uint16_t& __ctaMask)
{
  // __space == space_cluster (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
  if constexpr (__cta_group == cta_group_1)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group:"
        ":1 [%0], [%1, {%2, %3}], [%4], %5;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__smem_bar)),
          "h"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group:"
        ":2 [%0], [%1, {%2, %3}], [%4], %5;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__smem_bar)),
          "h"(__ctaMask)
        : "memory");
  }
}
#endif // __cccl_ptx_isa >= 860

/*
// cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group [dstMem], [tensorMap,
tensorCoords], [smem_bar], ctaMask; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a,
SM_110f
// .dst       = { .shared::cluster }
// .src       = { .global }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[3],
  uint64_t* smem_bar,
  const uint16_t& ctaMask);
*/
#if __cccl_ptx_isa >= 860
template <::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  ::cuda::ptx::space_cluster_t,
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group,
  void* __dstMem,
  const void* __tensorMap,
  const ::cuda::std::int32_t (&__tensorCoords)[3],
  ::cuda::std::uint64_t* __smem_bar,
  const ::cuda::std::uint16_t& __ctaMask)
{
  // __space == space_cluster (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
  if constexpr (__cta_group == cta_group_1)
  {
    asm("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group:"
        ":1 [%0], [%1, {%2, %3, %4}], [%5], %6;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__smem_bar)),
          "h"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2)
  {
    asm("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group:"
        ":2 [%0], [%1, {%2, %3, %4}], [%5], %6;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__smem_bar)),
          "h"(__ctaMask)
        : "memory");
  }
}
#endif // __cccl_ptx_isa >= 860

/*
// cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group [dstMem], [tensorMap,
tensorCoords], [smem_bar], ctaMask; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a,
SM_110f
// .dst       = { .shared::cluster }
// .src       = { .global }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[4],
  uint64_t* smem_bar,
  const uint16_t& ctaMask);
*/
#if __cccl_ptx_isa >= 860
template <::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  ::cuda::ptx::space_cluster_t,
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group,
  void* __dstMem,
  const void* __tensorMap,
  const ::cuda::std::int32_t (&__tensorCoords)[4],
  ::cuda::std::uint64_t* __smem_bar,
  const ::cuda::std::uint16_t& __ctaMask)
{
  // __space == space_cluster (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
  if constexpr (__cta_group == cta_group_1)
  {
    asm("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group:"
        ":1 [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__smem_bar)),
          "h"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2)
  {
    asm("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group:"
        ":2 [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__smem_bar)),
          "h"(__ctaMask)
        : "memory");
  }
}
#endif // __cccl_ptx_isa >= 860

/*
// cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group [dstMem], [tensorMap,
tensorCoords], [smem_bar], ctaMask; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a,
SM_110f
// .dst       = { .shared::cluster }
// .src       = { .global }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void cp_async_bulk_tensor(
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
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
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
    asm("cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group:"
        ":1 [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
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
    asm("cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group:"
        ":2 [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
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
// cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.multicast::cluster::32b.report_mechanism
[dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask; // PTX ISA 94, SM_107a, SM_107f
// .dst       = { .shared::cluster }
// .src       = { .global }
// .cta_group = { .cta_group::1, .cta_group::2 }
// .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000,
.mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80,
.mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff } template
<cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
__device__ static inline void cp_async_bulk_tensor_multicast_32b(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[1],
  uint64_t* smem_bar,
  const uint32_t& ctaMask);
*/
#if __cccl_ptx_isa >= 940
template <::cuda::ptx::dot_cta_group _Cta_Group, ::cuda::ptx::dot_report_mechanism _Report_Mechanism>
_CCCL_DEVICE static inline void cp_async_bulk_tensor_multicast_32b(
  ::cuda::ptx::space_cluster_t,
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group,
  ::cuda::ptx::report_mechanism_t<_Report_Mechanism> __report_mechanism,
  void* __dstMem,
  const void* __tensorMap,
  const ::cuda::std::int32_t (&__tensorCoords)[1],
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
    asm("cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::disabled [%0], [%1, {%2}], [%3], %4;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_80000000)
  {
    asm("cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80000000 [%0], [%1, {%2}], [%3], %4;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_8000)
  {
    asm("cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8000 [%0], [%1, {%2}], [%3], %4;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_80)
  {
    asm("cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80 [%0], [%1, {%2}], [%3], %4;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_8)
  {
    asm("cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8 [%0], [%1, {%2}], [%3], %4;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_element_ff)
  {
    asm("cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_element::ff [%0], [%1, {%2}], [%3], %4;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_disabled)
  {
    asm("cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::disabled [%0], [%1, {%2}], [%3], %4;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_80000000)
  {
    asm("cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80000000 [%0], [%1, {%2}], [%3], %4;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_8000)
  {
    asm("cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8000 [%0], [%1, {%2}], [%3], %4;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_80)
  {
    asm("cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80 [%0], [%1, {%2}], [%3], %4;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_8)
  {
    asm("cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8 [%0], [%1, {%2}], [%3], %4;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_element_ff)
  {
    asm("cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_element::ff [%0], [%1, {%2}], [%3], %4;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
}
#endif // __cccl_ptx_isa >= 940

/*
//
cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.multicast::cluster::32b.report_mechanism.override::global_address
[dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar], ctaMask; // PTX ISA 94, SM_107a, SM_107f
// .dst       = { .shared::cluster }
// .src       = { .global }
// .cta_group = { .cta_group::1, .cta_group::2 }
// .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000,
.mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80,
.mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff } template
<cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
__device__ static inline void cp_async_bulk_tensor_multicast_32b_override(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
  void* dstMem,
  const void* tensorMap,
  const void* gAddrToOverride,
  const int32_t (&tensorCoords)[1],
  uint64_t* smem_bar,
  const uint32_t& ctaMask);
*/
#if __cccl_ptx_isa >= 940
template <::cuda::ptx::dot_cta_group _Cta_Group, ::cuda::ptx::dot_report_mechanism _Report_Mechanism>
_CCCL_DEVICE static inline void cp_async_bulk_tensor_multicast_32b_override(
  ::cuda::ptx::space_cluster_t,
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group,
  ::cuda::ptx::report_mechanism_t<_Report_Mechanism> __report_mechanism,
  void* __dstMem,
  const void* __tensorMap,
  const void* __gAddrToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[1],
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
    asm("cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::disabled.override::global_address [%0], [%1, %2, {%3}], [%4], %5;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_80000000)
  {
    asm("cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80000000.override::global_address [%0], [%1, %2, {%3}], "
        "[%4], %5;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_8000)
  {
    asm("cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8000.override::global_address [%0], [%1, %2, {%3}], "
        "[%4], %5;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_80)
  {
    asm("cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80.override::global_address [%0], [%1, %2, {%3}], [%4], "
        "%5;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_8)
  {
    asm("cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8.override::global_address [%0], [%1, %2, {%3}], [%4], "
        "%5;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_element_ff)
  {
    asm("cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_element::ff.override::global_address [%0], [%1, %2, {%3}], [%4], "
        "%5;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_disabled)
  {
    asm("cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::disabled.override::global_address [%0], [%1, %2, {%3}], [%4], %5;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_80000000)
  {
    asm("cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80000000.override::global_address [%0], [%1, %2, {%3}], "
        "[%4], %5;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_8000)
  {
    asm("cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8000.override::global_address [%0], [%1, %2, {%3}], "
        "[%4], %5;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_80)
  {
    asm("cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80.override::global_address [%0], [%1, %2, {%3}], [%4], "
        "%5;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_8)
  {
    asm("cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8.override::global_address [%0], [%1, %2, {%3}], [%4], "
        "%5;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_element_ff)
  {
    asm("cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_element::ff.override::global_address [%0], [%1, %2, {%3}], [%4], "
        "%5;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
}
#endif // __cccl_ptx_isa >= 940

/*
//
cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.multicast::cluster::32b.report_mechanism.override::global_address.override::global_dim
[dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorCoords], [smem_bar], ctaMask; // PTX ISA 94, SM_107a,
SM_107f
// .dst       = { .shared::cluster }
// .src       = { .global }
// .cta_group = { .cta_group::1, .cta_group::2 }
// .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000,
.mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80,
.mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff } template <typename B16,
enable_if_t<sizeof(B16) == 2, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism
Report_Mechanism>
__device__ static inline void cp_async_bulk_tensor_multicast_32b_override(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
  void* dstMem,
  const void* tensorMap,
  const void* gAddrToOverride,
  const B16 (&tensorSizeToOverride)[1],
  const int32_t (&tensorCoords)[1],
  uint64_t* smem_bar,
  const uint32_t& ctaMask);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          ::cuda::ptx::dot_cta_group _Cta_Group,
          ::cuda::ptx::dot_report_mechanism _Report_Mechanism>
_CCCL_DEVICE static inline void cp_async_bulk_tensor_multicast_32b_override(
  ::cuda::ptx::space_cluster_t,
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group,
  ::cuda::ptx::report_mechanism_t<_Report_Mechanism> __report_mechanism,
  void* __dstMem,
  const void* __tensorMap,
  const void* __gAddrToOverride,
  const _B16 (&__tensorSizeToOverride)[1],
  const ::cuda::std::int32_t (&__tensorCoords)[1],
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
  static_assert(sizeof(_B16) == 2, "");
  if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_disabled)
  {
    asm("cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::disabled.override::global_address.override::global_dim [%0], [%1, %2, {%3}, "
        "{%4}], [%5], %6;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_80000000)
  {
    asm("cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80000000.override::global_address.override::global_dim "
        "[%0], [%1, %2, {%3}, {%4}], [%5], %6;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_8000)
  {
    asm("cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8000.override::global_address.override::global_dim "
        "[%0], [%1, %2, {%3}, {%4}], [%5], %6;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_80)
  {
    asm("cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80.override::global_address.override::global_dim [%0], "
        "[%1, %2, {%3}, {%4}], [%5], %6;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_8)
  {
    asm("cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8.override::global_address.override::global_dim [%0], "
        "[%1, %2, {%3}, {%4}], [%5], %6;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_element_ff)
  {
    asm("cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_element::ff.override::global_address.override::global_dim [%0], "
        "[%1, %2, {%3}, {%4}], [%5], %6;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_disabled)
  {
    asm("cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::disabled.override::global_address.override::global_dim [%0], [%1, %2, {%3}, "
        "{%4}], [%5], %6;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_80000000)
  {
    asm("cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80000000.override::global_address.override::global_dim "
        "[%0], [%1, %2, {%3}, {%4}], [%5], %6;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_8000)
  {
    asm("cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8000.override::global_address.override::global_dim "
        "[%0], [%1, %2, {%3}, {%4}], [%5], %6;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_80)
  {
    asm("cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80.override::global_address.override::global_dim [%0], "
        "[%1, %2, {%3}, {%4}], [%5], %6;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_8)
  {
    asm("cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8.override::global_address.override::global_dim [%0], "
        "[%1, %2, {%3}, {%4}], [%5], %6;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_element_ff)
  {
    asm("cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_element::ff.override::global_address.override::global_dim [%0], "
        "[%1, %2, {%3}, {%4}], [%5], %6;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
}
#endif // __cccl_ptx_isa >= 940

/*
// cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.multicast::cluster::32b.report_mechanism
[dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask; // PTX ISA 94, SM_107a, SM_107f
// .dst       = { .shared::cluster }
// .src       = { .global }
// .cta_group = { .cta_group::1, .cta_group::2 }
// .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000,
.mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80,
.mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff } template
<cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
__device__ static inline void cp_async_bulk_tensor_multicast_32b(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[2],
  uint64_t* smem_bar,
  const uint32_t& ctaMask);
*/
#if __cccl_ptx_isa >= 940
template <::cuda::ptx::dot_cta_group _Cta_Group, ::cuda::ptx::dot_report_mechanism _Report_Mechanism>
_CCCL_DEVICE static inline void cp_async_bulk_tensor_multicast_32b(
  ::cuda::ptx::space_cluster_t,
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group,
  ::cuda::ptx::report_mechanism_t<_Report_Mechanism> __report_mechanism,
  void* __dstMem,
  const void* __tensorMap,
  const ::cuda::std::int32_t (&__tensorCoords)[2],
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
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::disabled [%0], [%1, {%2, %3}], [%4], %5;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_80000000)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80000000 [%0], [%1, {%2, %3}], [%4], %5;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_8000)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8000 [%0], [%1, {%2, %3}], [%4], %5;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_80)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80 [%0], [%1, {%2, %3}], [%4], %5;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_8)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8 [%0], [%1, {%2, %3}], [%4], %5;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_element_ff)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_element::ff [%0], [%1, {%2, %3}], [%4], %5;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_disabled)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::disabled [%0], [%1, {%2, %3}], [%4], %5;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_80000000)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80000000 [%0], [%1, {%2, %3}], [%4], %5;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_8000)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8000 [%0], [%1, {%2, %3}], [%4], %5;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_80)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80 [%0], [%1, {%2, %3}], [%4], %5;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_8)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8 [%0], [%1, {%2, %3}], [%4], %5;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_element_ff)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_element::ff [%0], [%1, {%2, %3}], [%4], %5;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
}
#endif // __cccl_ptx_isa >= 940

/*
//
cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.multicast::cluster::32b.report_mechanism.override::global_address
[dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar], ctaMask; // PTX ISA 94, SM_107a, SM_107f
// .dst       = { .shared::cluster }
// .src       = { .global }
// .cta_group = { .cta_group::1, .cta_group::2 }
// .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000,
.mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80,
.mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff } template
<cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
__device__ static inline void cp_async_bulk_tensor_multicast_32b_override(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
  void* dstMem,
  const void* tensorMap,
  const void* gAddrToOverride,
  const int32_t (&tensorCoords)[2],
  uint64_t* smem_bar,
  const uint32_t& ctaMask);
*/
#if __cccl_ptx_isa >= 940
template <::cuda::ptx::dot_cta_group _Cta_Group, ::cuda::ptx::dot_report_mechanism _Report_Mechanism>
_CCCL_DEVICE static inline void cp_async_bulk_tensor_multicast_32b_override(
  ::cuda::ptx::space_cluster_t,
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group,
  ::cuda::ptx::report_mechanism_t<_Report_Mechanism> __report_mechanism,
  void* __dstMem,
  const void* __tensorMap,
  const void* __gAddrToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[2],
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
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::disabled.override::global_address [%0], [%1, %2, {%3, %4}], [%5], %6;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_80000000)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80000000.override::global_address [%0], [%1, %2, {%3, "
        "%4}], [%5], %6;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_8000)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8000.override::global_address [%0], [%1, %2, {%3, %4}], "
        "[%5], %6;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_80)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80.override::global_address [%0], [%1, %2, {%3, %4}], "
        "[%5], %6;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_8)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8.override::global_address [%0], [%1, %2, {%3, %4}], "
        "[%5], %6;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_element_ff)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_element::ff.override::global_address [%0], [%1, %2, {%3, %4}], "
        "[%5], %6;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_disabled)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::disabled.override::global_address [%0], [%1, %2, {%3, %4}], [%5], %6;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_80000000)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80000000.override::global_address [%0], [%1, %2, {%3, "
        "%4}], [%5], %6;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_8000)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8000.override::global_address [%0], [%1, %2, {%3, %4}], "
        "[%5], %6;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_80)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80.override::global_address [%0], [%1, %2, {%3, %4}], "
        "[%5], %6;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_8)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8.override::global_address [%0], [%1, %2, {%3, %4}], "
        "[%5], %6;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_element_ff)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_element::ff.override::global_address [%0], [%1, %2, {%3, %4}], "
        "[%5], %6;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
}
#endif // __cccl_ptx_isa >= 940

/*
//
cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.multicast::cluster::32b.report_mechanism.override::global_address.override::global_dim_stride
[dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride,
tensorCoords], [smem_bar], ctaMask; // PTX ISA 94, SM_107a, SM_107f
// .dst       = { .shared::cluster }
// .src       = { .global }
// .cta_group = { .cta_group::1, .cta_group::2 }
// .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000,
.mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80,
.mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff } template <typename B16,
enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true,
cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
__device__ static inline void cp_async_bulk_tensor_multicast_32b_override(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
  void* dstMem,
  const void* tensorMap,
  const void* gAddrToOverride,
  const B16 (&tensorSizeToOverride)[2],
  const B32 (&tensorLowerStrideToOverride)[1],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[2],
  uint64_t* smem_bar,
  const uint32_t& ctaMask);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true,
          ::cuda::ptx::dot_cta_group _Cta_Group,
          ::cuda::ptx::dot_report_mechanism _Report_Mechanism>
_CCCL_DEVICE static inline void cp_async_bulk_tensor_multicast_32b_override(
  ::cuda::ptx::space_cluster_t,
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group,
  ::cuda::ptx::report_mechanism_t<_Report_Mechanism> __report_mechanism,
  void* __dstMem,
  const void* __tensorMap,
  const void* __gAddrToOverride,
  const _B16 (&__tensorSizeToOverride)[2],
  const _B32 (&__tensorLowerStrideToOverride)[1],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[2],
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
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_disabled)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::disabled.override::global_address.override::global_dim_stride [%0], [%1, %2, "
        "{%3, %4}, {%5}, %6, {%7, %8}], [%9], %10;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_80000000)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80000000.override::global_address.override::global_dim_"
        "stride [%0], [%1, %2, {%3, %4}, {%5}, %6, {%7, %8}], [%9], %10;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_8000)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8000.override::global_address.override::global_dim_"
        "stride [%0], [%1, %2, {%3, %4}, {%5}, %6, {%7, %8}], [%9], %10;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_80)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80.override::global_address.override::global_dim_stride "
        "[%0], [%1, %2, {%3, %4}, {%5}, %6, {%7, %8}], [%9], %10;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_8)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8.override::global_address.override::global_dim_stride "
        "[%0], [%1, %2, {%3, %4}, {%5}, %6, {%7, %8}], [%9], %10;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_element_ff)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_element::ff.override::global_address.override::global_dim_stride "
        "[%0], [%1, %2, {%3, %4}, {%5}, %6, {%7, %8}], [%9], %10;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_disabled)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::disabled.override::global_address.override::global_dim_stride [%0], [%1, %2, "
        "{%3, %4}, {%5}, %6, {%7, %8}], [%9], %10;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_80000000)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80000000.override::global_address.override::global_dim_"
        "stride [%0], [%1, %2, {%3, %4}, {%5}, %6, {%7, %8}], [%9], %10;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_8000)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8000.override::global_address.override::global_dim_"
        "stride [%0], [%1, %2, {%3, %4}, {%5}, %6, {%7, %8}], [%9], %10;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_80)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80.override::global_address.override::global_dim_stride "
        "[%0], [%1, %2, {%3, %4}, {%5}, %6, {%7, %8}], [%9], %10;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_8)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8.override::global_address.override::global_dim_stride "
        "[%0], [%1, %2, {%3, %4}, {%5}, %6, {%7, %8}], [%9], %10;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_element_ff)
  {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_element::ff.override::global_address.override::global_dim_stride "
        "[%0], [%1, %2, {%3, %4}, {%5}, %6, {%7, %8}], [%9], %10;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
}
#endif // __cccl_ptx_isa >= 940

/*
// cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.multicast::cluster::32b.report_mechanism
[dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask; // PTX ISA 94, SM_107a, SM_107f
// .dst       = { .shared::cluster }
// .src       = { .global }
// .cta_group = { .cta_group::1, .cta_group::2 }
// .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000,
.mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80,
.mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff } template
<cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
__device__ static inline void cp_async_bulk_tensor_multicast_32b(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[3],
  uint64_t* smem_bar,
  const uint32_t& ctaMask);
*/
#if __cccl_ptx_isa >= 940
template <::cuda::ptx::dot_cta_group _Cta_Group, ::cuda::ptx::dot_report_mechanism _Report_Mechanism>
_CCCL_DEVICE static inline void cp_async_bulk_tensor_multicast_32b(
  ::cuda::ptx::space_cluster_t,
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group,
  ::cuda::ptx::report_mechanism_t<_Report_Mechanism> __report_mechanism,
  void* __dstMem,
  const void* __tensorMap,
  const ::cuda::std::int32_t (&__tensorCoords)[3],
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
    asm("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::disabled [%0], [%1, {%2, %3, %4}], [%5], %6;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_80000000)
  {
    asm("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80000000 [%0], [%1, {%2, %3, %4}], [%5], %6;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_8000)
  {
    asm("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8000 [%0], [%1, {%2, %3, %4}], [%5], %6;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_80)
  {
    asm("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80 [%0], [%1, {%2, %3, %4}], [%5], %6;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_8)
  {
    asm("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8 [%0], [%1, {%2, %3, %4}], [%5], %6;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_element_ff)
  {
    asm("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_element::ff [%0], [%1, {%2, %3, %4}], [%5], %6;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_disabled)
  {
    asm("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::disabled [%0], [%1, {%2, %3, %4}], [%5], %6;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_80000000)
  {
    asm("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80000000 [%0], [%1, {%2, %3, %4}], [%5], %6;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_8000)
  {
    asm("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8000 [%0], [%1, {%2, %3, %4}], [%5], %6;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_80)
  {
    asm("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80 [%0], [%1, {%2, %3, %4}], [%5], %6;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_8)
  {
    asm("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8 [%0], [%1, {%2, %3, %4}], [%5], %6;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_element_ff)
  {
    asm("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_element::ff [%0], [%1, {%2, %3, %4}], [%5], %6;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
}
#endif // __cccl_ptx_isa >= 940

/*
//
cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.multicast::cluster::32b.report_mechanism.override::global_address
[dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar], ctaMask; // PTX ISA 94, SM_107a, SM_107f
// .dst       = { .shared::cluster }
// .src       = { .global }
// .cta_group = { .cta_group::1, .cta_group::2 }
// .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000,
.mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80,
.mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff } template
<cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
__device__ static inline void cp_async_bulk_tensor_multicast_32b_override(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
  void* dstMem,
  const void* tensorMap,
  const void* gAddrToOverride,
  const int32_t (&tensorCoords)[3],
  uint64_t* smem_bar,
  const uint32_t& ctaMask);
*/
#if __cccl_ptx_isa >= 940
template <::cuda::ptx::dot_cta_group _Cta_Group, ::cuda::ptx::dot_report_mechanism _Report_Mechanism>
_CCCL_DEVICE static inline void cp_async_bulk_tensor_multicast_32b_override(
  ::cuda::ptx::space_cluster_t,
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group,
  ::cuda::ptx::report_mechanism_t<_Report_Mechanism> __report_mechanism,
  void* __dstMem,
  const void* __tensorMap,
  const void* __gAddrToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[3],
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
    asm("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::disabled.override::global_address [%0], [%1, %2, {%3, %4, %5}], [%6], %7;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_80000000)
  {
    asm("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80000000.override::global_address [%0], [%1, %2, {%3, "
        "%4, %5}], [%6], %7;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_8000)
  {
    asm("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8000.override::global_address [%0], [%1, %2, {%3, %4, "
        "%5}], [%6], %7;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_80)
  {
    asm("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80.override::global_address [%0], [%1, %2, {%3, %4, "
        "%5}], [%6], %7;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_8)
  {
    asm("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8.override::global_address [%0], [%1, %2, {%3, %4, "
        "%5}], [%6], %7;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_element_ff)
  {
    asm("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_element::ff.override::global_address [%0], [%1, %2, {%3, %4, "
        "%5}], [%6], %7;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_disabled)
  {
    asm("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::disabled.override::global_address [%0], [%1, %2, {%3, %4, %5}], [%6], %7;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_80000000)
  {
    asm("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80000000.override::global_address [%0], [%1, %2, {%3, "
        "%4, %5}], [%6], %7;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_8000)
  {
    asm("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8000.override::global_address [%0], [%1, %2, {%3, %4, "
        "%5}], [%6], %7;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_80)
  {
    asm("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80.override::global_address [%0], [%1, %2, {%3, %4, "
        "%5}], [%6], %7;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_8)
  {
    asm("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8.override::global_address [%0], [%1, %2, {%3, %4, "
        "%5}], [%6], %7;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_element_ff)
  {
    asm("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_element::ff.override::global_address [%0], [%1, %2, {%3, %4, "
        "%5}], [%6], %7;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
}
#endif // __cccl_ptx_isa >= 940

/*
//
cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.multicast::cluster::32b.report_mechanism.override::global_address.override::global_dim_stride
[dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride,
tensorCoords], [smem_bar], ctaMask; // PTX ISA 94, SM_107a, SM_107f
// .dst       = { .shared::cluster }
// .src       = { .global }
// .cta_group = { .cta_group::1, .cta_group::2 }
// .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000,
.mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80,
.mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff } template <typename B16,
enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true,
cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
__device__ static inline void cp_async_bulk_tensor_multicast_32b_override(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
  void* dstMem,
  const void* tensorMap,
  const void* gAddrToOverride,
  const B16 (&tensorSizeToOverride)[3],
  const B32 (&tensorLowerStrideToOverride)[2],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[3],
  uint64_t* smem_bar,
  const uint32_t& ctaMask);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true,
          ::cuda::ptx::dot_cta_group _Cta_Group,
          ::cuda::ptx::dot_report_mechanism _Report_Mechanism>
_CCCL_DEVICE static inline void cp_async_bulk_tensor_multicast_32b_override(
  ::cuda::ptx::space_cluster_t,
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group,
  ::cuda::ptx::report_mechanism_t<_Report_Mechanism> __report_mechanism,
  void* __dstMem,
  const void* __tensorMap,
  const void* __gAddrToOverride,
  const _B16 (&__tensorSizeToOverride)[3],
  const _B32 (&__tensorLowerStrideToOverride)[2],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[3],
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
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_disabled)
  {
    asm("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::disabled.override::global_address.override::global_dim_stride [%0], [%1, %2, "
        "{%3, %4, %5}, {%6, %7}, %8, {%9, %10, %11}], [%12], %13;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_80000000)
  {
    asm("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80000000.override::global_address.override::global_dim_"
        "stride [%0], [%1, %2, {%3, %4, %5}, {%6, %7}, %8, {%9, %10, %11}], [%12], %13;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_8000)
  {
    asm("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8000.override::global_address.override::global_dim_"
        "stride [%0], [%1, %2, {%3, %4, %5}, {%6, %7}, %8, {%9, %10, %11}], [%12], %13;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_80)
  {
    asm("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80.override::global_address.override::global_dim_stride "
        "[%0], [%1, %2, {%3, %4, %5}, {%6, %7}, %8, {%9, %10, %11}], [%12], %13;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_8)
  {
    asm("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8.override::global_address.override::global_dim_stride "
        "[%0], [%1, %2, {%3, %4, %5}, {%6, %7}, %8, {%9, %10, %11}], [%12], %13;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_element_ff)
  {
    asm("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_element::ff.override::global_address.override::global_dim_stride "
        "[%0], [%1, %2, {%3, %4, %5}, {%6, %7}, %8, {%9, %10, %11}], [%12], %13;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_disabled)
  {
    asm("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::disabled.override::global_address.override::global_dim_stride [%0], [%1, %2, "
        "{%3, %4, %5}, {%6, %7}, %8, {%9, %10, %11}], [%12], %13;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_80000000)
  {
    asm("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80000000.override::global_address.override::global_dim_"
        "stride [%0], [%1, %2, {%3, %4, %5}, {%6, %7}, %8, {%9, %10, %11}], [%12], %13;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_8000)
  {
    asm("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8000.override::global_address.override::global_dim_"
        "stride [%0], [%1, %2, {%3, %4, %5}, {%6, %7}, %8, {%9, %10, %11}], [%12], %13;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_80)
  {
    asm("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80.override::global_address.override::global_dim_stride "
        "[%0], [%1, %2, {%3, %4, %5}, {%6, %7}, %8, {%9, %10, %11}], [%12], %13;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_8)
  {
    asm("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8.override::global_address.override::global_dim_stride "
        "[%0], [%1, %2, {%3, %4, %5}, {%6, %7}, %8, {%9, %10, %11}], [%12], %13;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_element_ff)
  {
    asm("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_element::ff.override::global_address.override::global_dim_stride "
        "[%0], [%1, %2, {%3, %4, %5}, {%6, %7}, %8, {%9, %10, %11}], [%12], %13;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
}
#endif // __cccl_ptx_isa >= 940

/*
// cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.multicast::cluster::32b.report_mechanism
[dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask; // PTX ISA 94, SM_107a, SM_107f
// .dst       = { .shared::cluster }
// .src       = { .global }
// .cta_group = { .cta_group::1, .cta_group::2 }
// .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000,
.mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80,
.mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff } template
<cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
__device__ static inline void cp_async_bulk_tensor_multicast_32b(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[4],
  uint64_t* smem_bar,
  const uint32_t& ctaMask);
*/
#if __cccl_ptx_isa >= 940
template <::cuda::ptx::dot_cta_group _Cta_Group, ::cuda::ptx::dot_report_mechanism _Report_Mechanism>
_CCCL_DEVICE static inline void cp_async_bulk_tensor_multicast_32b(
  ::cuda::ptx::space_cluster_t,
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group,
  ::cuda::ptx::report_mechanism_t<_Report_Mechanism> __report_mechanism,
  void* __dstMem,
  const void* __tensorMap,
  const ::cuda::std::int32_t (&__tensorCoords)[4],
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
    asm("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::disabled [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_80000000)
  {
    asm("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80000000 [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_8000)
  {
    asm("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8000 [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_80)
  {
    asm("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80 [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_8)
  {
    asm("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8 [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_element_ff)
  {
    asm("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_element::ff [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_disabled)
  {
    asm("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::disabled [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_80000000)
  {
    asm("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80000000 [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_8000)
  {
    asm("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8000 [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_80)
  {
    asm("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80 [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_8)
  {
    asm("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8 [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_element_ff)
  {
    asm("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_element::ff [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
}
#endif // __cccl_ptx_isa >= 940

/*
//
cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.multicast::cluster::32b.report_mechanism.override::global_address
[dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar], ctaMask; // PTX ISA 94, SM_107a, SM_107f
// .dst       = { .shared::cluster }
// .src       = { .global }
// .cta_group = { .cta_group::1, .cta_group::2 }
// .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000,
.mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80,
.mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff } template
<cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
__device__ static inline void cp_async_bulk_tensor_multicast_32b_override(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
  void* dstMem,
  const void* tensorMap,
  const void* gAddrToOverride,
  const int32_t (&tensorCoords)[4],
  uint64_t* smem_bar,
  const uint32_t& ctaMask);
*/
#if __cccl_ptx_isa >= 940
template <::cuda::ptx::dot_cta_group _Cta_Group, ::cuda::ptx::dot_report_mechanism _Report_Mechanism>
_CCCL_DEVICE static inline void cp_async_bulk_tensor_multicast_32b_override(
  ::cuda::ptx::space_cluster_t,
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group,
  ::cuda::ptx::report_mechanism_t<_Report_Mechanism> __report_mechanism,
  void* __dstMem,
  const void* __tensorMap,
  const void* __gAddrToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[4],
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
    asm("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::disabled.override::global_address [%0], [%1, %2, {%3, %4, %5, %6}], [%7], %8;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_80000000)
  {
    asm("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80000000.override::global_address [%0], [%1, %2, {%3, "
        "%4, %5, %6}], [%7], %8;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_8000)
  {
    asm("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8000.override::global_address [%0], [%1, %2, {%3, %4, "
        "%5, %6}], [%7], %8;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_80)
  {
    asm("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80.override::global_address [%0], [%1, %2, {%3, %4, %5, "
        "%6}], [%7], %8;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_8)
  {
    asm("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8.override::global_address [%0], [%1, %2, {%3, %4, %5, "
        "%6}], [%7], %8;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_element_ff)
  {
    asm("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_element::ff.override::global_address [%0], [%1, %2, {%3, %4, %5, "
        "%6}], [%7], %8;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_disabled)
  {
    asm("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::disabled.override::global_address [%0], [%1, %2, {%3, %4, %5, %6}], [%7], %8;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_80000000)
  {
    asm("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80000000.override::global_address [%0], [%1, %2, {%3, "
        "%4, %5, %6}], [%7], %8;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_8000)
  {
    asm("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8000.override::global_address [%0], [%1, %2, {%3, %4, "
        "%5, %6}], [%7], %8;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_80)
  {
    asm("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80.override::global_address [%0], [%1, %2, {%3, %4, %5, "
        "%6}], [%7], %8;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_8)
  {
    asm("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8.override::global_address [%0], [%1, %2, {%3, %4, %5, "
        "%6}], [%7], %8;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_element_ff)
  {
    asm("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_element::ff.override::global_address [%0], [%1, %2, {%3, %4, %5, "
        "%6}], [%7], %8;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
}
#endif // __cccl_ptx_isa >= 940

/*
//
cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.multicast::cluster::32b.report_mechanism.override::global_address.override::global_dim_stride
[dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride,
tensorCoords], [smem_bar], ctaMask; // PTX ISA 94, SM_107a, SM_107f
// .dst       = { .shared::cluster }
// .src       = { .global }
// .cta_group = { .cta_group::1, .cta_group::2 }
// .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000,
.mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80,
.mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff } template <typename B16,
enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true,
cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
__device__ static inline void cp_async_bulk_tensor_multicast_32b_override(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
  void* dstMem,
  const void* tensorMap,
  const void* gAddrToOverride,
  const B16 (&tensorSizeToOverride)[4],
  const B32 (&tensorLowerStrideToOverride)[3],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[4],
  uint64_t* smem_bar,
  const uint32_t& ctaMask);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true,
          ::cuda::ptx::dot_cta_group _Cta_Group,
          ::cuda::ptx::dot_report_mechanism _Report_Mechanism>
_CCCL_DEVICE static inline void cp_async_bulk_tensor_multicast_32b_override(
  ::cuda::ptx::space_cluster_t,
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group,
  ::cuda::ptx::report_mechanism_t<_Report_Mechanism> __report_mechanism,
  void* __dstMem,
  const void* __tensorMap,
  const void* __gAddrToOverride,
  const _B16 (&__tensorSizeToOverride)[4],
  const _B32 (&__tensorLowerStrideToOverride)[3],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[4],
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
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_disabled)
  {
    asm("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::disabled.override::global_address.override::global_dim_stride [%0], [%1, %2, "
        "{%3, %4, %5, %6}, {%7, %8, %9}, %10, {%11, %12, %13, %14}], [%15], %16;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_80000000)
  {
    asm("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80000000.override::global_address.override::global_dim_"
        "stride [%0], [%1, %2, {%3, %4, %5, %6}, {%7, %8, %9}, %10, {%11, %12, %13, %14}], [%15], %16;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_8000)
  {
    asm("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8000.override::global_address.override::global_dim_"
        "stride [%0], [%1, %2, {%3, %4, %5, %6}, {%7, %8, %9}, %10, {%11, %12, %13, %14}], [%15], %16;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_80)
  {
    asm("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80.override::global_address.override::global_dim_stride "
        "[%0], [%1, %2, {%3, %4, %5, %6}, {%7, %8, %9}, %10, {%11, %12, %13, %14}], [%15], %16;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_8)
  {
    asm("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8.override::global_address.override::global_dim_stride "
        "[%0], [%1, %2, {%3, %4, %5, %6}, {%7, %8, %9}, %10, {%11, %12, %13, %14}], [%15], %16;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_element_ff)
  {
    asm("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_element::ff.override::global_address.override::global_dim_stride "
        "[%0], [%1, %2, {%3, %4, %5, %6}, {%7, %8, %9}, %10, {%11, %12, %13, %14}], [%15], %16;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_disabled)
  {
    asm("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::disabled.override::global_address.override::global_dim_stride [%0], [%1, %2, "
        "{%3, %4, %5, %6}, {%7, %8, %9}, %10, {%11, %12, %13, %14}], [%15], %16;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_80000000)
  {
    asm("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80000000.override::global_address.override::global_dim_"
        "stride [%0], [%1, %2, {%3, %4, %5, %6}, {%7, %8, %9}, %10, {%11, %12, %13, %14}], [%15], %16;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_8000)
  {
    asm("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8000.override::global_address.override::global_dim_"
        "stride [%0], [%1, %2, {%3, %4, %5, %6}, {%7, %8, %9}, %10, {%11, %12, %13, %14}], [%15], %16;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_80)
  {
    asm("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80.override::global_address.override::global_dim_stride "
        "[%0], [%1, %2, {%3, %4, %5, %6}, {%7, %8, %9}, %10, {%11, %12, %13, %14}], [%15], %16;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_8)
  {
    asm("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8.override::global_address.override::global_dim_stride "
        "[%0], [%1, %2, {%3, %4, %5, %6}, {%7, %8, %9}, %10, {%11, %12, %13, %14}], [%15], %16;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_element_ff)
  {
    asm("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_element::ff.override::global_address.override::global_dim_stride "
        "[%0], [%1, %2, {%3, %4, %5, %6}, {%7, %8, %9}, %10, {%11, %12, %13, %14}], [%15], %16;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__smem_bar)),
          "r"(__ctaMask)
        : "memory");
  }
}
#endif // __cccl_ptx_isa >= 940

/*
// cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.multicast::cluster::32b.report_mechanism
[dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask; // PTX ISA 94, SM_107a, SM_107f
// .dst       = { .shared::cluster }
// .src       = { .global }
// .cta_group = { .cta_group::1, .cta_group::2 }
// .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000,
.mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80,
.mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff } template
<cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
__device__ static inline void cp_async_bulk_tensor_multicast_32b(
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
_CCCL_DEVICE static inline void cp_async_bulk_tensor_multicast_32b(
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
    asm("cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::disabled [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
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
    asm("cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80000000 [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
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
    asm("cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8000 [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
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
    asm("cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80 [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
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
    asm("cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8 [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
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
    asm("cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_element::ff [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
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
    asm("cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::disabled [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
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
    asm("cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80000000 [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
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
    asm("cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8000 [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
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
    asm("cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80 [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
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
    asm("cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8 [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
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
    asm("cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_element::ff [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
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
cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.multicast::cluster::32b.report_mechanism.override::global_address
[dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar], ctaMask; // PTX ISA 94, SM_107a, SM_107f
// .dst       = { .shared::cluster }
// .src       = { .global }
// .cta_group = { .cta_group::1, .cta_group::2 }
// .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000,
.mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80,
.mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff } template
<cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
__device__ static inline void cp_async_bulk_tensor_multicast_32b_override(
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
_CCCL_DEVICE static inline void cp_async_bulk_tensor_multicast_32b_override(
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
    asm(
      "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
      "cluster::32b.mbarrier::report::disabled.override::global_address [%0], [%1, %2, {%3, %4, %5, %6, %7}], [%8], %9;"
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
    asm("cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80000000.override::global_address [%0], [%1, %2, {%3, "
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
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_8000)
  {
    asm("cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8000.override::global_address [%0], [%1, %2, {%3, %4, "
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
  else if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_valid_per_16bytes_80)
  {
    asm("cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80.override::global_address [%0], [%1, %2, {%3, %4, %5, "
        "%6, %7}], [%8], %9;"
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
    asm("cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8.override::global_address [%0], [%1, %2, {%3, %4, %5, "
        "%6, %7}], [%8], %9;"
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
    asm("cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_element::ff.override::global_address [%0], [%1, %2, {%3, %4, %5, "
        "%6, %7}], [%8], %9;"
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
    asm(
      "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
      "cluster::32b.mbarrier::report::disabled.override::global_address [%0], [%1, %2, {%3, %4, %5, %6, %7}], [%8], %9;"
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
    asm("cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80000000.override::global_address [%0], [%1, %2, {%3, "
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
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_8000)
  {
    asm("cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8000.override::global_address [%0], [%1, %2, {%3, %4, "
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
  else if constexpr (__cta_group == cta_group_2 && __report_mechanism == mbarrier_report_valid_per_16bytes_80)
  {
    asm("cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80.override::global_address [%0], [%1, %2, {%3, %4, %5, "
        "%6, %7}], [%8], %9;"
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
    asm("cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8.override::global_address [%0], [%1, %2, {%3, %4, %5, "
        "%6, %7}], [%8], %9;"
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
    asm("cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_element::ff.override::global_address [%0], [%1, %2, {%3, %4, %5, "
        "%6, %7}], [%8], %9;"
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
//
cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.multicast::cluster::32b.report_mechanism.override::global_address.override::global_dim_stride
[dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride,
tensorCoords], [smem_bar], ctaMask; // PTX ISA 94, SM_107a, SM_107f
// .dst       = { .shared::cluster }
// .src       = { .global }
// .cta_group = { .cta_group::1, .cta_group::2 }
// .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000,
.mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80,
.mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff } template <typename B16,
enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true,
cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
__device__ static inline void cp_async_bulk_tensor_multicast_32b_override(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
  void* dstMem,
  const void* tensorMap,
  const void* gAddrToOverride,
  const B16 (&tensorSizeToOverride)[5],
  const B32 (&tensorLowerStrideToOverride)[4],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[5],
  uint64_t* smem_bar,
  const uint32_t& ctaMask);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true,
          ::cuda::ptx::dot_cta_group _Cta_Group,
          ::cuda::ptx::dot_report_mechanism _Report_Mechanism>
_CCCL_DEVICE static inline void cp_async_bulk_tensor_multicast_32b_override(
  ::cuda::ptx::space_cluster_t,
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group,
  ::cuda::ptx::report_mechanism_t<_Report_Mechanism> __report_mechanism,
  void* __dstMem,
  const void* __tensorMap,
  const void* __gAddrToOverride,
  const _B16 (&__tensorSizeToOverride)[5],
  const _B32 (&__tensorLowerStrideToOverride)[4],
  const _B16& __tensorUpperStrideToOverride,
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
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  if constexpr (__cta_group == cta_group_1 && __report_mechanism == mbarrier_report_disabled)
  {
    asm("cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::disabled.override::global_address.override::global_dim_stride [%0], [%1, %2, "
        "{%3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, {%13, %14, %15, %16, %17}], [%18], %19;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[4])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[3])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
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
    asm("cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80000000.override::global_address.override::global_dim_"
        "stride [%0], [%1, %2, {%3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, {%13, %14, %15, %16, %17}], [%18], %19;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[4])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[3])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
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
    asm("cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8000.override::global_address.override::global_dim_"
        "stride [%0], [%1, %2, {%3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, {%13, %14, %15, %16, %17}], [%18], %19;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[4])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[3])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
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
    asm("cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80.override::global_address.override::global_dim_stride "
        "[%0], [%1, %2, {%3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, {%13, %14, %15, %16, %17}], [%18], %19;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[4])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[3])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
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
    asm("cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8.override::global_address.override::global_dim_stride "
        "[%0], [%1, %2, {%3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, {%13, %14, %15, %16, %17}], [%18], %19;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[4])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[3])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
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
    asm("cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1.multicast::"
        "cluster::32b.mbarrier::report::validity::per_element::ff.override::global_address.override::global_dim_stride "
        "[%0], [%1, %2, {%3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, {%13, %14, %15, %16, %17}], [%18], %19;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[4])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[3])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
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
    asm("cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::disabled.override::global_address.override::global_dim_stride [%0], [%1, %2, "
        "{%3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, {%13, %14, %15, %16, %17}], [%18], %19;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[4])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[3])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
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
    asm("cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80000000.override::global_address.override::global_dim_"
        "stride [%0], [%1, %2, {%3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, {%13, %14, %15, %16, %17}], [%18], %19;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[4])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[3])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
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
    asm("cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8000.override::global_address.override::global_dim_"
        "stride [%0], [%1, %2, {%3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, {%13, %14, %15, %16, %17}], [%18], %19;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[4])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[3])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
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
    asm("cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::80.override::global_address.override::global_dim_stride "
        "[%0], [%1, %2, {%3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, {%13, %14, %15, %16, %17}], [%18], %19;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[4])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[3])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
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
    asm("cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_16bytes::8.override::global_address.override::global_dim_stride "
        "[%0], [%1, %2, {%3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, {%13, %14, %15, %16, %17}], [%18], %19;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[4])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[3])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
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
    asm("cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::"
        "cluster::32b.mbarrier::report::validity::per_element::ff.override::global_address.override::global_dim_stride "
        "[%0], [%1, %2, {%3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, {%13, %14, %15, %16, %17}], [%18], %19;"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[4])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[3])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
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

#endif // _CUDA_PTX_GENERATED_CP_ASYNC_BULK_TENSOR_MULTICAST_H_
