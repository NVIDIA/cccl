//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <libnvcc/exception_filter.hpp>

#ifdef _WIN32
#  include <mutex>

// Deliberately confined to this TU: <windows.h> defines macros such as
// IMAGE_FILE_MACHINE_AMD64, which would break references like
// llvm::COFF::IMAGE_FILE_MACHINE_AMD64 in compiler.cpp.
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#endif

namespace libnvcc
{
#ifdef _WIN32
namespace
{
std::mutex g_filter_mutex;
int g_filter_depth                         = 0;
LPTOP_LEVEL_EXCEPTION_FILTER g_host_filter = nullptr;
} // namespace

UnhandledExceptionFilterGuard::UnhandledExceptionFilterGuard()
{
  const std::lock_guard<std::mutex> lock(g_filter_mutex);
  if (g_filter_depth++ == 0)
  {
    // Windows offers no way to read the current filter, so install a null one
    // and immediately put the observed value back.
    g_host_filter = SetUnhandledExceptionFilter(nullptr);
    SetUnhandledExceptionFilter(g_host_filter);
  }
}

UnhandledExceptionFilterGuard::~UnhandledExceptionFilterGuard()
{
  const std::lock_guard<std::mutex> lock(g_filter_mutex);
  if (--g_filter_depth == 0)
  {
    SetUnhandledExceptionFilter(g_host_filter);
    g_host_filter = nullptr;
  }
}
#else
UnhandledExceptionFilterGuard::UnhandledExceptionFilterGuard()  = default;
UnhandledExceptionFilterGuard::~UnhandledExceptionFilterGuard() = default;
#endif
} // namespace libnvcc
