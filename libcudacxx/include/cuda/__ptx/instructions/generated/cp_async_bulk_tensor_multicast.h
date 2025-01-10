// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_CP_ASYNC_BULK_TENSOR_MULTICAST_H_
#define _CUDA_PTX_GENERATED_CP_ASYNC_BULK_TENSOR_MULTICAST_H_

/*
// cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem], [tensorMap,
tensorCoords], [smem_bar], ctaMask; // 2a. PTX ISA 80, SM_90a
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
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90a__();
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  space_cluster_t,
  space_global_t,
  void* __dstMem,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[1],
  _CUDA_VSTD::uint64_t* __smem_bar,
  const _CUDA_VSTD::uint16_t& __ctaMask)
{
  // __space == space_cluster (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  NV_IF_ELSE_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (asm("cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [%0], "
         "[%1, {%2}], [%3], %4; // 2a." : : "r"(__as_ptr_smem(__dstMem)),
         "l"(__tensorMap),
         "r"(__tensorCoords[0]),
         "r"(__as_ptr_smem(__smem_bar)),
         "h"(__ctaMask) : "memory");),
    (
      // Unsupported architectures will have a linker error with a semi-decent error message
      __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90a__();));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem], [tensorMap,
tensorCoords], [smem_bar], ctaMask; // 2b. PTX ISA 80, SM_90a
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
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90a__();
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  space_cluster_t,
  space_global_t,
  void* __dstMem,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[2],
  _CUDA_VSTD::uint64_t* __smem_bar,
  const _CUDA_VSTD::uint16_t& __ctaMask)
{
  // __space == space_cluster (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  NV_IF_ELSE_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [%0], "
         "[%1, {%2, %3}], [%4], %5; // 2b." : : "r"(__as_ptr_smem(__dstMem)),
         "l"(__tensorMap),
         "r"(__tensorCoords[0]),
         "r"(__tensorCoords[1]),
         "r"(__as_ptr_smem(__smem_bar)),
         "h"(__ctaMask) : "memory");),
    (
      // Unsupported architectures will have a linker error with a semi-decent error message
      __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90a__();));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem], [tensorMap,
tensorCoords], [smem_bar], ctaMask; // 2c. PTX ISA 80, SM_90a
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
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90a__();
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  space_cluster_t,
  space_global_t,
  void* __dstMem,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[3],
  _CUDA_VSTD::uint64_t* __smem_bar,
  const _CUDA_VSTD::uint16_t& __ctaMask)
{
  // __space == space_cluster (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  NV_IF_ELSE_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (asm("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [%0], "
         "[%1, {%2, %3, %4}], [%5], %6; // 2c." : : "r"(__as_ptr_smem(__dstMem)),
         "l"(__tensorMap),
         "r"(__tensorCoords[0]),
         "r"(__tensorCoords[1]),
         "r"(__tensorCoords[2]),
         "r"(__as_ptr_smem(__smem_bar)),
         "h"(__ctaMask) : "memory");),
    (
      // Unsupported architectures will have a linker error with a semi-decent error message
      __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90a__();));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem], [tensorMap,
tensorCoords], [smem_bar], ctaMask; // 2d. PTX ISA 80, SM_90a
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
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90a__();
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  space_cluster_t,
  space_global_t,
  void* __dstMem,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[4],
  _CUDA_VSTD::uint64_t* __smem_bar,
  const _CUDA_VSTD::uint16_t& __ctaMask)
{
  // __space == space_cluster (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  NV_IF_ELSE_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (asm("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [%0], "
         "[%1, {%2, %3, %4, %5}], [%6], %7; // 2d." : : "r"(__as_ptr_smem(__dstMem)),
         "l"(__tensorMap),
         "r"(__tensorCoords[0]),
         "r"(__tensorCoords[1]),
         "r"(__tensorCoords[2]),
         "r"(__tensorCoords[3]),
         "r"(__as_ptr_smem(__smem_bar)),
         "h"(__ctaMask) : "memory");),
    (
      // Unsupported architectures will have a linker error with a semi-decent error message
      __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90a__();));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem], [tensorMap,
tensorCoords], [smem_bar], ctaMask; // 2e. PTX ISA 80, SM_90a
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
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90a__();
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  space_cluster_t,
  space_global_t,
  void* __dstMem,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[5],
  _CUDA_VSTD::uint64_t* __smem_bar,
  const _CUDA_VSTD::uint16_t& __ctaMask)
{
  // __space == space_cluster (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  NV_IF_ELSE_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (asm("cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [%0], "
         "[%1, {%2, %3, %4, %5, %6}], [%7], %8; // 2e." : : "r"(__as_ptr_smem(__dstMem)),
         "l"(__tensorMap),
         "r"(__tensorCoords[0]),
         "r"(__tensorCoords[1]),
         "r"(__tensorCoords[2]),
         "r"(__tensorCoords[3]),
         "r"(__tensorCoords[4]),
         "r"(__as_ptr_smem(__smem_bar)),
         "h"(__ctaMask) : "memory");),
    (
      // Unsupported architectures will have a linker error with a semi-decent error message
      __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90a__();));
}
#endif // __cccl_ptx_isa >= 800

#endif // _CUDA_PTX_GENERATED_CP_ASYNC_BULK_TENSOR_MULTICAST_H_
