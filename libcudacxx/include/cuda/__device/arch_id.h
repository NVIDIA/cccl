//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___DEVICE_ARCH_ID_H
#define _CUDA___DEVICE_ARCH_ID_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__fwd/devices.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

inline constexpr int __arch_specific_id_multiplier = 100000;

//! @brief Architecture identifier
//! This type identifies an architecture. It has more possible entries than just numeric values of the compute
//! capability. For example, sm_90 and sm_90a have the same compute capability, but the identifier is different.
enum class arch_id : int
{
  sm_60   = 60,
  sm_61   = 61,
  sm_70   = 70,
  sm_75   = 75,
  sm_80   = 80,
  sm_86   = 86,
  sm_89   = 89,
  sm_90   = 90,
  sm_100  = 100,
  sm_103  = 103,
  sm_110  = 110,
  sm_120  = 120,
  sm_90a  = 90 * __arch_specific_id_multiplier,
  sm_100a = 100 * __arch_specific_id_multiplier,
  sm_103a = 103 * __arch_specific_id_multiplier,
  sm_110a = 110 * __arch_specific_id_multiplier,
  sm_120a = 120 * __arch_specific_id_multiplier,
};

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___DEVICE_ARCH_ID_H
