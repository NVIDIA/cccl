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

#ifndef _CUDA_PTX_GETCTARANK_H_
#define _CUDA_PTX_GETCTARANK_H_

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

// 9.7.8.23. Data Movement and Conversion Instructions: getctarank
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-getctarank
/*
// getctarank{.space}.u32 dest, addr; // PTX ISA 78, SM_90
// .space     = { .shared::cluster }
template <typename=void>
__device__ static inline uint32_t getctarank(
  cuda::ptx::space_cluster_t,
  const void* addr);
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_getctarank_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t getctarank(
  space_cluster_t,
  const void* __addr)
{
  // __space == space_cluster (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    _CUDA_VSTD::uint32_t __dest;
    asm (
      "getctarank.shared::cluster.u32 %0, %1;"
      : "=r"(__dest)
      : "r"(__as_ptr_smem(__addr))
      :
    );
    return __dest;
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_getctarank_is_not_supported_before_SM_90__();
    return 0;
  ));
}
#endif // __cccl_ptx_isa >= 780

_LIBCUDACXX_END_NAMESPACE_CUDA_PTX

#endif // _CUDA_PTX_GETCTARANK_H_
