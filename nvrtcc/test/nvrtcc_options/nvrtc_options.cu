//===----------------------------------------------------------------------===//
//
// Part of nvrtcc in CUDA Core Compute Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include "../common/check_predefined_macros.h"

#if defined(__CUDACC_RTC__) && defined(__NV_BUILTIN_MOVE_FORWARD)
#  error "-builtin-move-forward=false was not passed properly"
#endif // __CUDACC_RTC__ && __NV_BUILTIN_MOVE_FORWARD
#if defined(__CUDACC_RTC__) && defined(__NV_BUILTIN_INITIALIZER_LIST)
#  error "-builtin-initializer-list=false was not passed properly"
#endif // __CUDACC_RTC__ && __NV_BUILTIN_INITIALIZER_LIST
#if defined(NVRTC_ONLY_MACRO) != defined(__CUDACC_RTC__)
#  error "-DNVRTC_ONLY_MACRO was not passed properly"
#endif // NVRTC_ONLY_MACRO != __CUDACC_RTC__

#if defined(NVRTC_ONLY_MACRO)
static_assert(NVRTC_ONLY_MACRO == 42);
#endif // NVRTC_ONLY_MACRO
