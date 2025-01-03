//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/__cccl/os.h>

#if _CCCL_OS(WINDOWS)
#  include <windows.h>
constexpr int windows = 1;
#else
constexpr int windows = 0;
#endif

#if _CCCL_OS(LINUX)
#  include <unistd.h>
constexpr int linux_ = 1;
#else
constexpr int linux_ = 0;
#endif

#if _CCCL_OS(ANDROID)
#  include <android/api-level.h>
constexpr int android = 1;
#endif

#if _CCCL_OS(QNX)
#  include <qnx.h>
constexpr int qnx = 1;
#endif

int main(int, char**)
{
  static_assert(windows + linux_ == 1, "");
#if _CCCL_OS(ANDROID) || _CCCL_OS(QNX)
  static_assert(linux_ == 1, "");
  static_assert(android + qnx == 1, "");
#endif
#if _CCCL_OS(LINUX)
  static_assert(windows == 0, "");
#endif
#if _CCCL_OS(WINDOWS)
  static_assert(android + qnx + linux_ == 0, "");
#endif
  return 0;
}
