//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/__cccl/os.h>

#if !defined(__CUDACC_RTC__)
#  if _CCCL_OS(WINDOWS)
#    include <windows.h>
#  endif

#  if _CCCL_OS(LINUX)
#    include <unistd.h>
#  endif

#  if _CCCL_OS(ANDROID)
#    include <android/api-level.h>
#  endif

#  if _CCCL_OS(QNX)
#    include <qnx.h>
#  endif
#endif

int main(int, char**)
{
  static_assert(_CCCL_OS(WINDOWS) + _CCCL_OS(LINUX) == 1, "");
#if _CCCL_OS(ANDROID) || _CCCL_OS(QNX)
  static_assert(_CCCL_OS(LINUX) == 1, "");
  static_assert(_CCCL_OS(ANDROID) + _CCCL_OS(QNX) == 1, "");
#endif
#if _CCCL_OS(LINUX)
  static_assert(_CCCL_OS(WINDOWS) == 0, "");
#endif
#if _CCCL_OS(WINDOWS)
  static_assert(_CCCL_OS(ANDROID) + _CCCL_OS(QNX) + _CCCL_OS(LINUX) == 0, "");
#endif
  return 0;
}
