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

#ifndef _CUDA_PTX_ST_ASYNC_H_
#define _CUDA_PTX_ST_ASYNC_H_

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

// 9.7.8.12. Data Movement and Conversion Instructions: st.async
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st-async
/*
// st.async.weak.shared::cluster.mbarrier::complete_tx::bytes{.type} [addr], value, [remote_bar];    // 1.  PTX ISA 81, SM_90
// .type      = { .b32, .b64 }
template <typename Type>
__device__ static inline void st_async(
  Type* addr,
  const Type& value,
  uint64_t* remote_bar);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_st_async_is_not_supported_before_SM_90__();
template <typename _Type>
_CCCL_DEVICE static inline void st_async(
  _Type* __addr,
  const _Type& __value,
  _CUDA_VSTD::uint64_t* __remote_bar)
{
  static_assert(sizeof(_Type) == 4 || sizeof(_Type) == 8, "");
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (sizeof(_Type) == 4) {
      asm (
        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];    // 1. "
        :
        : "r"(__as_ptr_remote_dsmem(__addr)),
          "r"(__as_b32(__value)),
          "r"(__as_ptr_remote_dsmem(__remote_bar))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (sizeof(_Type) == 8) {
      asm (
        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.b64 [%0], %1, [%2];    // 1. "
        :
        : "r"(__as_ptr_remote_dsmem(__addr)),
          "l"(__as_b64(__value)),
          "r"(__as_ptr_remote_dsmem(__remote_bar))
        : "memory"
      );
    }
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_st_async_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 810

/*
// st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v2{.type} [addr], value, [remote_bar]; // 2.  PTX ISA 81, SM_90
// .type      = { .b32, .b64 }
template <typename Type>
__device__ static inline void st_async(
  Type* addr,
  const Type (&value)[2],
  uint64_t* remote_bar);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_st_async_is_not_supported_before_SM_90__();
template <typename _Type>
_CCCL_DEVICE static inline void st_async(
  _Type* __addr,
  const _Type (&__value)[2],
  _CUDA_VSTD::uint64_t* __remote_bar)
{
  static_assert(sizeof(_Type) == 4 || sizeof(_Type) == 8, "");
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (sizeof(_Type) == 4) {
      asm (
        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3]; // 2. "
        :
        : "r"(__as_ptr_remote_dsmem(__addr)),
          "r"(__as_b32(__value[0])),
          "r"(__as_b32(__value[1])),
          "r"(__as_ptr_remote_dsmem(__remote_bar))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (sizeof(_Type) == 8) {
      asm (
        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v2.b64 [%0], {%1, %2}, [%3]; // 2. "
        :
        : "r"(__as_ptr_remote_dsmem(__addr)),
          "l"(__as_b64(__value[0])),
          "l"(__as_b64(__value[1])),
          "r"(__as_ptr_remote_dsmem(__remote_bar))
        : "memory"
      );
    }
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_st_async_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 810

/*
// st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [addr], value, [remote_bar];    // 3.  PTX ISA 81, SM_90
template <typename B32>
__device__ static inline void st_async(
  B32* addr,
  const B32 (&value)[4],
  uint64_t* remote_bar);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_st_async_is_not_supported_before_SM_90__();
template <typename _B32>
_CCCL_DEVICE static inline void st_async(
  _B32* __addr,
  const _B32 (&__value)[4],
  _CUDA_VSTD::uint64_t* __remote_bar)
{
  static_assert(sizeof(_B32) == 4, "");
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];    // 3. "
      :
      : "r"(__as_ptr_remote_dsmem(__addr)),
        "r"(__as_b32(__value[0])),
        "r"(__as_b32(__value[1])),
        "r"(__as_b32(__value[2])),
        "r"(__as_b32(__value[3])),
        "r"(__as_ptr_remote_dsmem(__remote_bar))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_st_async_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 810

_LIBCUDACXX_END_NAMESPACE_CUDA_PTX

#endif // _CUDA_PTX_ST_ASYNC_H_
