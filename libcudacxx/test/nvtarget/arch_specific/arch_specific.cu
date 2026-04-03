//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// This test checks if arch-specific NV target macros work properly.

#include <nv/target>

// Currently, nvcc is the only compiler that supports arch-specific architectures.
#if !defined(__NVCC__)
#  error "This test works with nvcc only."
#endif // !__NVCC__

#if defined(__CUDA_ARCH__)
#  if __CUDA_ARCH_SPECIFIC__ != 1030
#    error "This test must be compiled for sm_103a target."
#  endif // __CUDA_ARCH_SPECIFIC__ != 1030
#endif // __CUDA_ARCH__

#define CHECK_TRUE(_PRED)                                                 \
  do                                                                      \
  {                                                                       \
    NV_IF_ELSE_TARGET(_PRED, static_assert(true);, static_assert(false);) \
  } while (0)
#define CHECK_FALSE(_PRED)                                                \
  do                                                                      \
  {                                                                       \
    NV_IF_ELSE_TARGET(_PRED, static_assert(false);, static_assert(true);) \
  } while (0)

__host__ __device__ void fn()
{
#if defined(__CUDA_ARCH__)
  CHECK_TRUE(NV_IS_EXACTLY_SM_103);

  CHECK_TRUE(NV_HAS_FEATURE_SM_103a);

  CHECK_TRUE(NV_HAS_FEATURE_SM_100f);
  CHECK_TRUE(NV_HAS_FEATURE_SM_103f);
#else // ^^^ __CUDA_ARCH__ ^^^ / vvv !__CUDA_ARCH__ vvv
  CHECK_TRUE(NV_IS_HOST);

  CHECK_FALSE(NV_HAS_FEATURE_SM_103a);

  CHECK_FALSE(NV_HAS_FEATURE_SM_100f);
  CHECK_FALSE(NV_HAS_FEATURE_SM_103f);
#endif // ^^^ !__CUDA_ARCH__ ^^^

  CHECK_FALSE(NV_HAS_FEATURE_SM_100a);
  CHECK_FALSE(NV_HAS_FEATURE_SM_110a);

  CHECK_FALSE(NV_HAS_FEATURE_SM_110f);
}
