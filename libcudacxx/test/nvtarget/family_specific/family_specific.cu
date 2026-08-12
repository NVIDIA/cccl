//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// This test checks if family-specific NV target macros work properly.

#include <nv/target>

// Currently, nvcc is the only compiler that supports arch-specific architectures.
#if !defined(__NVCC__)
#  error "This test works with nvcc only."
#endif // !__NVCC__

#if defined(__CUDA_ARCH_FAMILY_SPECIFIC__)
#  if __CUDA_ARCH_FAMILY_SPECIFIC__ != 1070
#    error "This test must be compiled for sm_107f target."
#  endif // __CUDA_ARCH_FAMILY_SPECIFIC__ != 1070
#endif // __CUDA_ARCH_FAMILY_SPECIFIC__

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

#ifdef __CUDACC_TILE__
__tile__
#endif // __CUDACC_TILE__
  __host__ __device__ void
  fn()
{
#if defined(__CUDA_ARCH_FAMILY_SPECIFIC__)
  CHECK_TRUE(NV_IS_EXACTLY_SM_107);

  CHECK_TRUE(NV_HAS_FEATURE_SM_100f);
  CHECK_TRUE(NV_HAS_FEATURE_SM_103f);
  CHECK_TRUE(NV_HAS_FEATURE_SM_107f);
#elif !defined(__CUDA_ARCH__) // ^^^ __CUDA_ARCH_FAMILY_SPECIFIC__ ^^^ / vvv host vvv
  CHECK_TRUE(NV_IS_HOST);

  CHECK_FALSE(NV_HAS_FEATURE_SM_100f);
  CHECK_FALSE(NV_HAS_FEATURE_SM_103f);
  CHECK_FALSE(NV_HAS_FEATURE_SM_107f);
#endif // ^^^ host ^^^

  CHECK_FALSE(NV_HAS_FEATURE_SM_110f);

  CHECK_FALSE(NV_HAS_FEATURE_SM_100a);
  CHECK_FALSE(NV_HAS_FEATURE_SM_103a);
  CHECK_FALSE(NV_HAS_FEATURE_SM_107a);
  CHECK_FALSE(NV_HAS_FEATURE_SM_110a);
}
