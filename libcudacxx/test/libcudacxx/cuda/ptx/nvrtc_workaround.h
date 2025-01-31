//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// NVRTC ships a built-in copy of <nv/detail/__target_macros>, so including CCCL's version of this header will omit the
// content since the header guards are already defined. To make older NVRTC versions have a few newer feature macros
// required for the PTX tests, we define them here outside the header guards.
// TODO(bgruber): limit this workaround to NVRTC versions older than the first one shipping those macros
#ifdef __CUDACC_RTC__
#  ifndef NV_HAS_FEATURE_SM_100a
#    define NV_HAS_FEATURE_SM_100a __NV_HAS_FEATURE_SM_100a
#    if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && defined(__CUDA_ARCH_FEAT_SM100_ALL))
#      define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_100a 1
#    else
#      define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_100a 0
#    endif
#  endif // NV_HAS_FEATURE_SM_100a

// Re-enable sm_101a support in nvcc.
#  ifndef NV_HAS_FEATURE_SM_101a
#    define NV_HAS_FEATURE_SM_101a __NV_HAS_FEATURE_SM_101a
#    if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1010) && defined(__CUDA_ARCH_FEAT_SM101_ALL))
#      define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_101a 1
#    else
#      define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_101a 0
#    endif
#  endif // NV_HAS_FEATURE_SM_101a
#endif // __CUDACC_RTC__
