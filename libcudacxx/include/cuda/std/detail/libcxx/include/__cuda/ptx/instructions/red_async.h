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

#ifndef _CUDA_PTX_RED_ASYNC_H_
#define _CUDA_PTX_RED_ASYNC_H_

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

// 9.7.12.7. Parallel Synchronization and Communication Instructions: red.async
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-red-async
/*
// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
// .type      = { .u32 }
// .op        = { .inc }
template <typename=void>
__device__ static inline void red_async(
  cuda::ptx::op_inc_t,
  uint32_t* dest,
  const uint32_t& value,
  uint64_t* remote_bar);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void red_async(
  op_inc_t,
  _CUDA_VSTD::uint32_t* __dest,
  const _CUDA_VSTD::uint32_t& __value,
  _CUDA_VSTD::uint64_t* __remote_bar)
{
  // __type == type_u32 (due to parameter type constraint)
  // __op == op_inc (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.inc.u32  [%0], %1, [%2]; "
      :
      : "r"(__as_ptr_remote_dsmem(__dest)),
        "r"(__value),
        "r"(__as_ptr_remote_dsmem(__remote_bar))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_red_async_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 810

/*
// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
// .type      = { .u32 }
// .op        = { .dec }
template <typename=void>
__device__ static inline void red_async(
  cuda::ptx::op_dec_t,
  uint32_t* dest,
  const uint32_t& value,
  uint64_t* remote_bar);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void red_async(
  op_dec_t,
  _CUDA_VSTD::uint32_t* __dest,
  const _CUDA_VSTD::uint32_t& __value,
  _CUDA_VSTD::uint64_t* __remote_bar)
{
  // __type == type_u32 (due to parameter type constraint)
  // __op == op_dec (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.dec.u32  [%0], %1, [%2]; "
      :
      : "r"(__as_ptr_remote_dsmem(__dest)),
        "r"(__value),
        "r"(__as_ptr_remote_dsmem(__remote_bar))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_red_async_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 810

/*
// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
// .type      = { .u32 }
// .op        = { .min }
template <typename=void>
__device__ static inline void red_async(
  cuda::ptx::op_min_t,
  uint32_t* dest,
  const uint32_t& value,
  uint64_t* remote_bar);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void red_async(
  op_min_t,
  _CUDA_VSTD::uint32_t* __dest,
  const _CUDA_VSTD::uint32_t& __value,
  _CUDA_VSTD::uint64_t* __remote_bar)
{
  // __type == type_u32 (due to parameter type constraint)
  // __op == op_min (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.min.u32  [%0], %1, [%2]; "
      :
      : "r"(__as_ptr_remote_dsmem(__dest)),
        "r"(__value),
        "r"(__as_ptr_remote_dsmem(__remote_bar))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_red_async_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 810

/*
// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
// .type      = { .u32 }
// .op        = { .max }
template <typename=void>
__device__ static inline void red_async(
  cuda::ptx::op_max_t,
  uint32_t* dest,
  const uint32_t& value,
  uint64_t* remote_bar);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void red_async(
  op_max_t,
  _CUDA_VSTD::uint32_t* __dest,
  const _CUDA_VSTD::uint32_t& __value,
  _CUDA_VSTD::uint64_t* __remote_bar)
{
  // __type == type_u32 (due to parameter type constraint)
  // __op == op_max (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.max.u32  [%0], %1, [%2]; "
      :
      : "r"(__as_ptr_remote_dsmem(__dest)),
        "r"(__value),
        "r"(__as_ptr_remote_dsmem(__remote_bar))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_red_async_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 810

/*
// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
// .type      = { .u32 }
// .op        = { .add }
template <typename=void>
__device__ static inline void red_async(
  cuda::ptx::op_add_t,
  uint32_t* dest,
  const uint32_t& value,
  uint64_t* remote_bar);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void red_async(
  op_add_t,
  _CUDA_VSTD::uint32_t* __dest,
  const _CUDA_VSTD::uint32_t& __value,
  _CUDA_VSTD::uint64_t* __remote_bar)
{
  // __type == type_u32 (due to parameter type constraint)
  // __op == op_add (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.add.u32  [%0], %1, [%2]; "
      :
      : "r"(__as_ptr_remote_dsmem(__dest)),
        "r"(__value),
        "r"(__as_ptr_remote_dsmem(__remote_bar))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_red_async_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 810

/*
// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
// .type      = { .s32 }
// .op        = { .min }
template <typename=void>
__device__ static inline void red_async(
  cuda::ptx::op_min_t,
  int32_t* dest,
  const int32_t& value,
  uint64_t* remote_bar);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void red_async(
  op_min_t,
  _CUDA_VSTD::int32_t* __dest,
  const _CUDA_VSTD::int32_t& __value,
  _CUDA_VSTD::uint64_t* __remote_bar)
{
  // __type == type_s32 (due to parameter type constraint)
  // __op == op_min (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.min.s32  [%0], %1, [%2]; "
      :
      : "r"(__as_ptr_remote_dsmem(__dest)),
        "r"(__value),
        "r"(__as_ptr_remote_dsmem(__remote_bar))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_red_async_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 810

/*
// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
// .type      = { .s32 }
// .op        = { .max }
template <typename=void>
__device__ static inline void red_async(
  cuda::ptx::op_max_t,
  int32_t* dest,
  const int32_t& value,
  uint64_t* remote_bar);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void red_async(
  op_max_t,
  _CUDA_VSTD::int32_t* __dest,
  const _CUDA_VSTD::int32_t& __value,
  _CUDA_VSTD::uint64_t* __remote_bar)
{
  // __type == type_s32 (due to parameter type constraint)
  // __op == op_max (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.max.s32  [%0], %1, [%2]; "
      :
      : "r"(__as_ptr_remote_dsmem(__dest)),
        "r"(__value),
        "r"(__as_ptr_remote_dsmem(__remote_bar))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_red_async_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 810

/*
// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
// .type      = { .s32 }
// .op        = { .add }
template <typename=void>
__device__ static inline void red_async(
  cuda::ptx::op_add_t,
  int32_t* dest,
  const int32_t& value,
  uint64_t* remote_bar);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void red_async(
  op_add_t,
  _CUDA_VSTD::int32_t* __dest,
  const _CUDA_VSTD::int32_t& __value,
  _CUDA_VSTD::uint64_t* __remote_bar)
{
  // __type == type_s32 (due to parameter type constraint)
  // __op == op_add (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.add.s32  [%0], %1, [%2]; "
      :
      : "r"(__as_ptr_remote_dsmem(__dest)),
        "r"(__value),
        "r"(__as_ptr_remote_dsmem(__remote_bar))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_red_async_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 810

/*
// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
// .type      = { .b32 }
// .op        = { .and }
template <typename B32>
__device__ static inline void red_async(
  cuda::ptx::op_and_op_t,
  B32* dest,
  const B32& value,
  uint64_t* remote_bar);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename _B32>
_CCCL_DEVICE static inline void red_async(
  op_and_op_t,
  _B32* __dest,
  const _B32& __value,
  _CUDA_VSTD::uint64_t* __remote_bar)
{
  // __type == type_b32 (due to parameter type constraint)
  // __op == op_and_op (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.and.b32  [%0], %1, [%2]; "
      :
      : "r"(__as_ptr_remote_dsmem(__dest)),
        "r"(__as_b32(__value)),
        "r"(__as_ptr_remote_dsmem(__remote_bar))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_red_async_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 810

/*
// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
// .type      = { .b32 }
// .op        = { .or }
template <typename B32>
__device__ static inline void red_async(
  cuda::ptx::op_or_op_t,
  B32* dest,
  const B32& value,
  uint64_t* remote_bar);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename _B32>
_CCCL_DEVICE static inline void red_async(
  op_or_op_t,
  _B32* __dest,
  const _B32& __value,
  _CUDA_VSTD::uint64_t* __remote_bar)
{
  // __type == type_b32 (due to parameter type constraint)
  // __op == op_or_op (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.or.b32  [%0], %1, [%2]; "
      :
      : "r"(__as_ptr_remote_dsmem(__dest)),
        "r"(__as_b32(__value)),
        "r"(__as_ptr_remote_dsmem(__remote_bar))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_red_async_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 810

/*
// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
// .type      = { .b32 }
// .op        = { .xor }
template <typename B32>
__device__ static inline void red_async(
  cuda::ptx::op_xor_op_t,
  B32* dest,
  const B32& value,
  uint64_t* remote_bar);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename _B32>
_CCCL_DEVICE static inline void red_async(
  op_xor_op_t,
  _B32* __dest,
  const _B32& __value,
  _CUDA_VSTD::uint64_t* __remote_bar)
{
  // __type == type_b32 (due to parameter type constraint)
  // __op == op_xor_op (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.xor.b32  [%0], %1, [%2]; "
      :
      : "r"(__as_ptr_remote_dsmem(__dest)),
        "r"(__as_b32(__value)),
        "r"(__as_ptr_remote_dsmem(__remote_bar))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_red_async_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 810

/*
// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
// .type      = { .u64 }
// .op        = { .add }
template <typename=void>
__device__ static inline void red_async(
  cuda::ptx::op_add_t,
  uint64_t* dest,
  const uint64_t& value,
  uint64_t* remote_bar);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void red_async(
  op_add_t,
  _CUDA_VSTD::uint64_t* __dest,
  const _CUDA_VSTD::uint64_t& __value,
  _CUDA_VSTD::uint64_t* __remote_bar)
{
  // __type == type_u64 (due to parameter type constraint)
  // __op == op_add (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.add.u64  [%0], %1, [%2]; "
      :
      : "r"(__as_ptr_remote_dsmem(__dest)),
        "l"(__value),
        "r"(__as_ptr_remote_dsmem(__remote_bar))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_red_async_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 810

/*
// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}.u64  [dest], value, [remote_bar]; // .u64 intentional PTX ISA 81, SM_90
// .op        = { .add }
template <typename=void>
__device__ static inline void red_async(
  cuda::ptx::op_add_t,
  int64_t* dest,
  const int64_t& value,
  int64_t* remote_bar);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void red_async(
  op_add_t,
  _CUDA_VSTD::int64_t* __dest,
  const _CUDA_VSTD::int64_t& __value,
  _CUDA_VSTD::int64_t* __remote_bar)
{
  // __op == op_add (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.add.u64  [%0], %1, [%2]; // .u64 intentional"
      :
      : "r"(__as_ptr_remote_dsmem(__dest)),
        "l"(__value),
        "r"(__as_ptr_remote_dsmem(__remote_bar))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_red_async_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 810

_LIBCUDACXX_END_NAMESPACE_CUDA_PTX

#endif // _CUDA_PTX_RED_ASYNC_H_
