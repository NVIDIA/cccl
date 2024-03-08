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

#ifndef _CUDA_PTX_CP_ASYNC_BULK_WAIT_GROUP_H_
#define _CUDA_PTX_CP_ASYNC_BULK_WAIT_GROUP_H_

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

// 9.7.8.24.13. Data Movement and Conversion Instructions: cp.async.bulk.wait_group
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group
/*
// cp.async.bulk.wait_group N; // PTX ISA 80, SM_90
template <int N32>
__device__ static inline void cp_async_bulk_wait_group(
  cuda::ptx::n32_t<N32> N);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_wait_group_is_not_supported_before_SM_90__();
template <int _N32>
_CCCL_DEVICE static inline void cp_async_bulk_wait_group(
  n32_t<_N32> __N)
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm volatile (
      "cp.async.bulk.wait_group %0;"
      :
      : "n"(__N)
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_cp_async_bulk_wait_group_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.wait_group.read N; // PTX ISA 80, SM_90
template <int N32>
__device__ static inline void cp_async_bulk_wait_group_read(
  cuda::ptx::n32_t<N32> N);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_wait_group_read_is_not_supported_before_SM_90__();
template <int _N32>
_CCCL_DEVICE static inline void cp_async_bulk_wait_group_read(
  n32_t<_N32> __N)
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm volatile (
      "cp.async.bulk.wait_group.read %0;"
      :
      : "n"(__N)
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_cp_async_bulk_wait_group_read_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

_LIBCUDACXX_END_NAMESPACE_CUDA_PTX

#endif // _CUDA_PTX_CP_ASYNC_BULK_WAIT_GROUP_H_
