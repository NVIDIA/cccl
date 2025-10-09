//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CCCL_TARGET_MACROS_H
#define __CCCL_TARGET_MACROS_H

#include <cuda/std/__cccl/compiler.h>
#include <cuda/std/__cccl/system_header.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/dialect.h>

#include <nv/target>

#if _CCCL_CUDA_COMPILER(NVHPC)
#  define _CCCL_TARGET(X) target(X)
#elif _CCCL_DEVICE_COMPILATION()
__host__ __device__ _CCCL_CONSTEVAL ::nv::target::detail::base_int_t
__cccl_ptx_arch_to_nv_target_sm_mask(int __ptx_arch) noexcept
{
  switch (__ptx_arch)
  {
    case 350:
      return ::nv::target::detail::sm_35_bit;
    case 370:
      return ::nv::target::detail::sm_37_bit;
    case 500:
      return ::nv::target::detail::sm_50_bit;
    case 520:
      return ::nv::target::detail::sm_52_bit;
    case 530:
      return ::nv::target::detail::sm_53_bit;
    case 600:
      return ::nv::target::detail::sm_60_bit;
    case 610:
      return ::nv::target::detail::sm_61_bit;
    case 620:
      return ::nv::target::detail::sm_62_bit;
    case 700:
      return ::nv::target::detail::sm_70_bit;
    case 720:
      return ::nv::target::detail::sm_72_bit;
    case 750:
      return ::nv::target::detail::sm_75_bit;
    case 800:
      return ::nv::target::detail::sm_80_bit;
    case 860:
      return ::nv::target::detail::sm_86_bit;
    case 870:
      return ::nv::target::detail::sm_87_bit;
    case 890:
      return ::nv::target::detail::sm_89_bit;
    case 900:
      return ::nv::target::detail::sm_90_bit;
    case 1000:
      return ::nv::target::detail::sm_100_bit;
    case 1030:
      return ::nv::target::detail::sm_103_bit;
    case 1100:
      return ::nv::target::detail::sm_110_bit;
    case 1200:
      return ::nv::target::detail::sm_120_bit;
    default:
      _CCCL_UNREACHABLE();
  }
}
#  define _CCCL_TARGET(X) constexpr((X).targets & __cccl_ptx_arch_to_nv_target_sm_mask(__CUDA_ARCH__))
#else
#  define _CCCL_TARGET(X) constexpr((X).targets & nv::target::detail::all_hosts)
#endif

#endif // __CCCL_TARGET_MACROS_H
