// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_PTX_CP_ASYNC_BULK_TENSOR_H_
#define _CUDA_PTX_CP_ASYNC_BULK_TENSOR_H_

#ifndef __cuda_std__
#  include <__config>
#endif // __cuda_std__

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <nv/target> // __CUDA_MINIMUM_ARCH__ and friends

#include "../ptx_dot_variants.h"
#include "../ptx_helper_functions.h"
#include "../../../cstdint"

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_PTX

// 9.7.8.24.9. Data Movement and Conversion Instructions: cp.async.bulk.tensor
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor
/*
// cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap, tensorCoords], [smem_bar];// 1a. PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .global }
template <typename=void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[1],
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  space_cluster_t,
  space_global_t,
  void* __dstMem,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[1],
  _CUDA_VSTD::uint64_t* __smem_bar)
{
  // __space == space_cluster (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2}], [%3];// 1a."
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.1d.dst.src.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 3a. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
template <typename=void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  const void* tensorMap,
  const int32_t (&tensorCoords)[1],
  const void* srcMem);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  space_global_t,
  space_shared_t,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[1],
  const void* __srcMem)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "cp.async.bulk.tensor.1d.global.shared::cta.tile.bulk_group [%0, {%1}], [%2]; // 3a."
      :
      : "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__as_ptr_smem(__srcMem))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap, tensorCoords], [smem_bar];// 1b. PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .global }
template <typename=void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[2],
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  space_cluster_t,
  space_global_t,
  void* __dstMem,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[2],
  _CUDA_VSTD::uint64_t* __smem_bar)
{
  // __space == space_cluster (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];// 1b."
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.2d.dst.src.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 3b. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
template <typename=void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  const void* tensorMap,
  const int32_t (&tensorCoords)[2],
  const void* srcMem);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  space_global_t,
  space_shared_t,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[2],
  const void* __srcMem)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%1, %2}], [%3]; // 3b."
      :
      : "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__as_ptr_smem(__srcMem))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap, tensorCoords], [smem_bar];// 1c. PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .global }
template <typename=void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[3],
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  space_cluster_t,
  space_global_t,
  void* __dstMem,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[3],
  _CUDA_VSTD::uint64_t* __smem_bar)
{
  // __space == space_cluster (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];// 1c."
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.3d.dst.src.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 3c. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
template <typename=void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  const void* tensorMap,
  const int32_t (&tensorCoords)[3],
  const void* srcMem);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  space_global_t,
  space_shared_t,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[3],
  const void* __srcMem)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group [%0, {%1, %2, %3}], [%4]; // 3c."
      :
      : "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__as_ptr_smem(__srcMem))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap, tensorCoords], [smem_bar];// 1d. PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .global }
template <typename=void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[4],
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  space_cluster_t,
  space_global_t,
  void* __dstMem,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[4],
  _CUDA_VSTD::uint64_t* __smem_bar)
{
  // __space == space_cluster (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5}], [%6];// 1d."
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.4d.dst.src.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 3d. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
template <typename=void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  const void* tensorMap,
  const int32_t (&tensorCoords)[4],
  const void* srcMem);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  space_global_t,
  space_shared_t,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[4],
  const void* __srcMem)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group [%0, {%1, %2, %3, %4}], [%5]; // 3d."
      :
      : "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__as_ptr_smem(__srcMem))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap, tensorCoords], [smem_bar];// 1e. PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .global }
template <typename=void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[5],
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  space_cluster_t,
  space_global_t,
  void* __dstMem,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[5],
  _CUDA_VSTD::uint64_t* __smem_bar)
{
  // __space == space_cluster (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5, %6}], [%7];// 1e."
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__tensorCoords[4]),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.5d.dst.src.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 3e. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
template <typename=void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  const void* tensorMap,
  const int32_t (&tensorCoords)[5],
  const void* srcMem);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  space_global_t,
  space_shared_t,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[5],
  const void* __srcMem)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group [%0, {%1, %2, %3, %4, %5}], [%6]; // 3e."
      :
      : "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__tensorCoords[4]),
        "r"(__as_ptr_smem(__srcMem))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800
/*
// cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask; // 2a. PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .global }
template <typename=void>
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
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
template <typename=void>
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
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2}], [%3], %4; // 2a."
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__as_ptr_smem(__smem_bar)),
        "h"(__ctaMask)
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask; // 2b. PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .global }
template <typename=void>
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
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
template <typename=void>
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
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, %3}], [%4], %5; // 2b."
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__as_ptr_smem(__smem_bar)),
        "h"(__ctaMask)
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask; // 2c. PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .global }
template <typename=void>
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
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
template <typename=void>
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
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, %3, %4}], [%5], %6; // 2c."
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__as_ptr_smem(__smem_bar)),
        "h"(__ctaMask)
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask; // 2d. PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .global }
template <typename=void>
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
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
template <typename=void>
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
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, %3, %4, %5}], [%6], %7; // 2d."
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__as_ptr_smem(__smem_bar)),
        "h"(__ctaMask)
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask; // 2e. PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .global }
template <typename=void>
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
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
template <typename=void>
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
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8; // 2e."
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
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

_LIBCUDACXX_END_NAMESPACE_CUDA_PTX

#endif // _CUDA_PTX_CP_ASYNC_BULK_TENSOR_H_
