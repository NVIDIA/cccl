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

#ifndef _CUDA_PTX_ELECT_SYNC_H_
#define _CUDA_PTX_ELECT_SYNC_H_

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/cstdint>
#include <cuda/std/detail/libcxx/include/__cuda/ptx/ptx_dot_variants.h>
#include <cuda/std/detail/libcxx/include/__cuda/ptx/ptx_helper_functions.h>

#include <nv/target> // __CUDA_MINIMUM_ARCH__ and friends

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_PTX

// elect_sync
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-elect-sync
/*
// elect.sync _|is_elected, membermask; // PTX ISA 80, SM_90
template <typename=void>
__device__ static inline bool elect_sync(
  const uint32_t& membermask);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_elect_sync_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline bool elect_sync(const _CUDA_VSTD::uint32_t& __membermask)
{
  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_90,
    (_CUDA_VSTD::uint32_t __is_elected; asm volatile(
       "{\n\t .reg .pred P_OUT; \n\t"
       "elect.sync _|P_OUT, %1;\n\t"
       "selp.b32 %0, 1, 0, P_OUT; \n"
       "}"
       : "=r"(__is_elected)
       : "r"(__membermask)
       :);
     return static_cast<bool>(__is_elected);),
    (
      // Unsupported architectures will have a linker error with a semi-decent error message
      __cuda_ptx_elect_sync_is_not_supported_before_SM_90__(); return false;));
}
#endif // __cccl_ptx_isa >= 800

_LIBCUDACXX_END_NAMESPACE_CUDA_PTX

#endif // _CUDA_PTX_ELECT_SYNC_H_
