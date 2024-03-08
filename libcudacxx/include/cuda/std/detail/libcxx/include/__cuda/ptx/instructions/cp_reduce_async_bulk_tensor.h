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

#ifndef _CUDA_PTX_CP_REDUCE_ASYNC_BULK_TENSOR_H_
#define _CUDA_PTX_CP_REDUCE_ASYNC_BULK_TENSOR_H_

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

// 9.7.8.24.10. Data Movement and Conversion Instructions: cp.reduce.async.bulk.tensor
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-reduce-async-bulk-tensor
/*
// cp.reduce.async.bulk.tensor.1d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1a. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
template <cuda::ptx::dot_op Op>
__device__ static inline void cp_reduce_async_bulk_tensor(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_t<Op> op,
  const void* tensorMap,
  const int32_t (&tensorCoords)[1],
  const void* srcMem);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_tensor_is_not_supported_before_SM_90__();
template <dot_op _Op>
_CCCL_DEVICE static inline void cp_reduce_async_bulk_tensor(
  space_global_t,
  space_shared_t,
  op_t<_Op> __op,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[1],
  const void* __srcMem)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  static_assert(__op == op_add || __op == op_min || __op == op_max || __op == op_inc || __op == op_dec || __op == op_and_op || __op == op_or_op || __op == op_xor_op, "");
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_add) {
      asm (
        "cp.reduce.async.bulk.tensor.1d.global.shared::cta.add.tile.bulk_group [%0, {%1}], [%2]; // 1a."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_min) {
      asm (
        "cp.reduce.async.bulk.tensor.1d.global.shared::cta.min.tile.bulk_group [%0, {%1}], [%2]; // 1a."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_max) {
      asm (
        "cp.reduce.async.bulk.tensor.1d.global.shared::cta.max.tile.bulk_group [%0, {%1}], [%2]; // 1a."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_inc) {
      asm (
        "cp.reduce.async.bulk.tensor.1d.global.shared::cta.inc.tile.bulk_group [%0, {%1}], [%2]; // 1a."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_dec) {
      asm (
        "cp.reduce.async.bulk.tensor.1d.global.shared::cta.dec.tile.bulk_group [%0, {%1}], [%2]; // 1a."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_and_op) {
      asm (
        "cp.reduce.async.bulk.tensor.1d.global.shared::cta.and.tile.bulk_group [%0, {%1}], [%2]; // 1a."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_or_op) {
      asm (
        "cp.reduce.async.bulk.tensor.1d.global.shared::cta.or.tile.bulk_group [%0, {%1}], [%2]; // 1a."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_xor_op) {
      asm (
        "cp.reduce.async.bulk.tensor.1d.global.shared::cta.xor.tile.bulk_group [%0, {%1}], [%2]; // 1a."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    }
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_cp_reduce_async_bulk_tensor_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.reduce.async.bulk.tensor.2d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1b. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
template <cuda::ptx::dot_op Op>
__device__ static inline void cp_reduce_async_bulk_tensor(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_t<Op> op,
  const void* tensorMap,
  const int32_t (&tensorCoords)[2],
  const void* srcMem);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_tensor_is_not_supported_before_SM_90__();
template <dot_op _Op>
_CCCL_DEVICE static inline void cp_reduce_async_bulk_tensor(
  space_global_t,
  space_shared_t,
  op_t<_Op> __op,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[2],
  const void* __srcMem)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  static_assert(__op == op_add || __op == op_min || __op == op_max || __op == op_inc || __op == op_dec || __op == op_and_op || __op == op_or_op || __op == op_xor_op, "");
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_add) {
      asm (
        "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group [%0, {%1, %2}], [%3]; // 1b."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_min) {
      asm (
        "cp.reduce.async.bulk.tensor.2d.global.shared::cta.min.tile.bulk_group [%0, {%1, %2}], [%3]; // 1b."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_max) {
      asm (
        "cp.reduce.async.bulk.tensor.2d.global.shared::cta.max.tile.bulk_group [%0, {%1, %2}], [%3]; // 1b."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_inc) {
      asm (
        "cp.reduce.async.bulk.tensor.2d.global.shared::cta.inc.tile.bulk_group [%0, {%1, %2}], [%3]; // 1b."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_dec) {
      asm (
        "cp.reduce.async.bulk.tensor.2d.global.shared::cta.dec.tile.bulk_group [%0, {%1, %2}], [%3]; // 1b."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_and_op) {
      asm (
        "cp.reduce.async.bulk.tensor.2d.global.shared::cta.and.tile.bulk_group [%0, {%1, %2}], [%3]; // 1b."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_or_op) {
      asm (
        "cp.reduce.async.bulk.tensor.2d.global.shared::cta.or.tile.bulk_group [%0, {%1, %2}], [%3]; // 1b."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_xor_op) {
      asm (
        "cp.reduce.async.bulk.tensor.2d.global.shared::cta.xor.tile.bulk_group [%0, {%1, %2}], [%3]; // 1b."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    }
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_cp_reduce_async_bulk_tensor_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.reduce.async.bulk.tensor.3d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1c. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
template <cuda::ptx::dot_op Op>
__device__ static inline void cp_reduce_async_bulk_tensor(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_t<Op> op,
  const void* tensorMap,
  const int32_t (&tensorCoords)[3],
  const void* srcMem);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_tensor_is_not_supported_before_SM_90__();
template <dot_op _Op>
_CCCL_DEVICE static inline void cp_reduce_async_bulk_tensor(
  space_global_t,
  space_shared_t,
  op_t<_Op> __op,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[3],
  const void* __srcMem)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  static_assert(__op == op_add || __op == op_min || __op == op_max || __op == op_inc || __op == op_dec || __op == op_and_op || __op == op_or_op || __op == op_xor_op, "");
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_add) {
      asm (
        "cp.reduce.async.bulk.tensor.3d.global.shared::cta.add.tile.bulk_group [%0, {%1, %2, %3}], [%4]; // 1c."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_min) {
      asm (
        "cp.reduce.async.bulk.tensor.3d.global.shared::cta.min.tile.bulk_group [%0, {%1, %2, %3}], [%4]; // 1c."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_max) {
      asm (
        "cp.reduce.async.bulk.tensor.3d.global.shared::cta.max.tile.bulk_group [%0, {%1, %2, %3}], [%4]; // 1c."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_inc) {
      asm (
        "cp.reduce.async.bulk.tensor.3d.global.shared::cta.inc.tile.bulk_group [%0, {%1, %2, %3}], [%4]; // 1c."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_dec) {
      asm (
        "cp.reduce.async.bulk.tensor.3d.global.shared::cta.dec.tile.bulk_group [%0, {%1, %2, %3}], [%4]; // 1c."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_and_op) {
      asm (
        "cp.reduce.async.bulk.tensor.3d.global.shared::cta.and.tile.bulk_group [%0, {%1, %2, %3}], [%4]; // 1c."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_or_op) {
      asm (
        "cp.reduce.async.bulk.tensor.3d.global.shared::cta.or.tile.bulk_group [%0, {%1, %2, %3}], [%4]; // 1c."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_xor_op) {
      asm (
        "cp.reduce.async.bulk.tensor.3d.global.shared::cta.xor.tile.bulk_group [%0, {%1, %2, %3}], [%4]; // 1c."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    }
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_cp_reduce_async_bulk_tensor_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.reduce.async.bulk.tensor.4d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1d. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
template <cuda::ptx::dot_op Op>
__device__ static inline void cp_reduce_async_bulk_tensor(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_t<Op> op,
  const void* tensorMap,
  const int32_t (&tensorCoords)[4],
  const void* srcMem);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_tensor_is_not_supported_before_SM_90__();
template <dot_op _Op>
_CCCL_DEVICE static inline void cp_reduce_async_bulk_tensor(
  space_global_t,
  space_shared_t,
  op_t<_Op> __op,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[4],
  const void* __srcMem)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  static_assert(__op == op_add || __op == op_min || __op == op_max || __op == op_inc || __op == op_dec || __op == op_and_op || __op == op_or_op || __op == op_xor_op, "");
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_add) {
      asm (
        "cp.reduce.async.bulk.tensor.4d.global.shared::cta.add.tile.bulk_group [%0, {%1, %2, %3, %4}], [%5]; // 1d."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_min) {
      asm (
        "cp.reduce.async.bulk.tensor.4d.global.shared::cta.min.tile.bulk_group [%0, {%1, %2, %3, %4}], [%5]; // 1d."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_max) {
      asm (
        "cp.reduce.async.bulk.tensor.4d.global.shared::cta.max.tile.bulk_group [%0, {%1, %2, %3, %4}], [%5]; // 1d."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_inc) {
      asm (
        "cp.reduce.async.bulk.tensor.4d.global.shared::cta.inc.tile.bulk_group [%0, {%1, %2, %3, %4}], [%5]; // 1d."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_dec) {
      asm (
        "cp.reduce.async.bulk.tensor.4d.global.shared::cta.dec.tile.bulk_group [%0, {%1, %2, %3, %4}], [%5]; // 1d."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_and_op) {
      asm (
        "cp.reduce.async.bulk.tensor.4d.global.shared::cta.and.tile.bulk_group [%0, {%1, %2, %3, %4}], [%5]; // 1d."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_or_op) {
      asm (
        "cp.reduce.async.bulk.tensor.4d.global.shared::cta.or.tile.bulk_group [%0, {%1, %2, %3, %4}], [%5]; // 1d."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_xor_op) {
      asm (
        "cp.reduce.async.bulk.tensor.4d.global.shared::cta.xor.tile.bulk_group [%0, {%1, %2, %3, %4}], [%5]; // 1d."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    }
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_cp_reduce_async_bulk_tensor_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.reduce.async.bulk.tensor.5d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1e. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
template <cuda::ptx::dot_op Op>
__device__ static inline void cp_reduce_async_bulk_tensor(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_t<Op> op,
  const void* tensorMap,
  const int32_t (&tensorCoords)[5],
  const void* srcMem);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_tensor_is_not_supported_before_SM_90__();
template <dot_op _Op>
_CCCL_DEVICE static inline void cp_reduce_async_bulk_tensor(
  space_global_t,
  space_shared_t,
  op_t<_Op> __op,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[5],
  const void* __srcMem)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  static_assert(__op == op_add || __op == op_min || __op == op_max || __op == op_inc || __op == op_dec || __op == op_and_op || __op == op_or_op || __op == op_xor_op, "");
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_add) {
      asm (
        "cp.reduce.async.bulk.tensor.5d.global.shared::cta.add.tile.bulk_group [%0, {%1, %2, %3, %4, %5}], [%6]; // 1e."
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
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_min) {
      asm (
        "cp.reduce.async.bulk.tensor.5d.global.shared::cta.min.tile.bulk_group [%0, {%1, %2, %3, %4, %5}], [%6]; // 1e."
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
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_max) {
      asm (
        "cp.reduce.async.bulk.tensor.5d.global.shared::cta.max.tile.bulk_group [%0, {%1, %2, %3, %4, %5}], [%6]; // 1e."
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
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_inc) {
      asm (
        "cp.reduce.async.bulk.tensor.5d.global.shared::cta.inc.tile.bulk_group [%0, {%1, %2, %3, %4, %5}], [%6]; // 1e."
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
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_dec) {
      asm (
        "cp.reduce.async.bulk.tensor.5d.global.shared::cta.dec.tile.bulk_group [%0, {%1, %2, %3, %4, %5}], [%6]; // 1e."
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
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_and_op) {
      asm (
        "cp.reduce.async.bulk.tensor.5d.global.shared::cta.and.tile.bulk_group [%0, {%1, %2, %3, %4, %5}], [%6]; // 1e."
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
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_or_op) {
      asm (
        "cp.reduce.async.bulk.tensor.5d.global.shared::cta.or.tile.bulk_group [%0, {%1, %2, %3, %4, %5}], [%6]; // 1e."
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
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_xor_op) {
      asm (
        "cp.reduce.async.bulk.tensor.5d.global.shared::cta.xor.tile.bulk_group [%0, {%1, %2, %3, %4, %5}], [%6]; // 1e."
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
    }
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_cp_reduce_async_bulk_tensor_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

_LIBCUDACXX_END_NAMESPACE_CUDA_PTX

#endif // _CUDA_PTX_CP_REDUCE_ASYNC_BULK_TENSOR_H_
