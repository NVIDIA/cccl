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

// missing SM_100
#  define _NV_TARGET_VAL_SM_100             1000
#  define _NV_TARGET___NV_PROVIDES_SM_100   (_NV_TARGET_PROVIDES(_NV_TARGET_VAL_SM_100))
#  define _NV_TARGET___NV_IS_EXACTLY_SM_100 (_NV_TARGET_IS_EXACTLY(_NV_TARGET_VAL_SM_100))
#  define NV_PROVIDES_SM_100                __NV_PROVIDES_SM_100
#  define NV_IS_EXACTLY_SM_100              __NV_IS_EXACTLY_SM_100
#  if (_NV_TARGET___NV_IS_EXACTLY_SM_100)
#    define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_100 1
#  else
#    define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_100 0
#  endif
#  if (_NV_TARGET___NV_PROVIDES_SM_100)
#    define _NV_TARGET_BOOL___NV_PROVIDES_SM_100 1
#  else
#    define _NV_TARGET_BOOL___NV_PROVIDES_SM_100 0
#  endif

#endif // __CUDACC_RTC__
