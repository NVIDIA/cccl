// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/cstdlib>

#if _CCCL_HOSTED()
#  include <cstdarg>
#  include <cstdio>
#endif // _CCCL_HOSTED()

CUB_NAMESPACE_BEGIN
namespace detail
{
//! Returns if logging is enabled
_CCCL_HOST_API inline bool logging_enabled() noexcept
{
#if _CCCL_HOSTED()
  static bool enabled = ::std::getenv("CCCL_EXPERIMENTAL_LOGGING") != nullptr;
  return enabled;
#else // _CCCL_HOSTED()
  return false;
#endif // _CCCL_HOSTED()
}

//! Logs the message, independently of whether logging is enabled
_CCCL_HOST_API inline void log_always([[maybe_unused]] const char* fmt, ...) noexcept
{
#if _CCCL_HOSTED()
  va_list args;
  va_start(args, fmt);
  ::vprintf(fmt, args);
  va_end(args);
#endif // _CCCL_HOSTED()
}

//! Logs the message when logging is enabled
_CCCL_HOST_API inline void log(const char* fmt, ...) noexcept
{
  if (logging_enabled())
  {
#if _CCCL_HOSTED()
    va_list args;
    va_start(args, fmt);
    ::vprintf(fmt, args);
    va_end(args);
#endif // _CCCL_HOSTED()
  }
}
} // namespace detail
CUB_NAMESPACE_END
