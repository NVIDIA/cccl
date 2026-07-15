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

#include <cuda/std/__host_stdlib/cstdarg>
#include <cuda/std/__host_stdlib/cstdio>
#include <cuda/std/cstdlib>

#ifdef _CCCL_DOXYGEN_INVOKED
//! When defined, disables all logging code in CCCL
#  define CCCL_DISABLE_LOGGING
#endif // _CCCL_DOXYGEN_INVOKED

CUB_NAMESPACE_BEGIN
namespace detail
{
//! Returns if logging is enabled via the CCCL_EXPERIMENTAL_LOGGING env variable (always false in device code)
[[nodiscard]] _CCCL_HOST_DEVICE_API inline bool logging_enabled() noexcept
{
#if _CCCL_HOSTED() && !defined(CCCL_DISABLE_LOGGING)
  NV_IF_TARGET(NV_IS_HOST,
               ({
                 static const bool enabled = [] {
                   const char* const env = ::std::getenv("CCCL_EXPERIMENTAL_LOGGING");
                   return env != nullptr && ::std::atoi(env) != 0;
                 }();
                 return enabled;
               }),
               ({ return false; }));
#else // _CCCL_HOSTED()
  return false;
#endif // _CCCL_HOSTED()
}

//! Logs the message when called from host code, independently of whether logging is enabled
_CCCL_ATTRIBUTE_FORMAT(__printf__, 1, 2)
_CCCL_HOST_DEVICE_API inline void log_always([[maybe_unused]] const char* fmt, ...) noexcept
{
#if _CCCL_HOSTED() && !defined(CCCL_DISABLE_LOGGING)
  NV_IF_TARGET(NV_IS_HOST, ({
                 ::std::va_list args;
                 va_start(args, fmt);
                 ::vprintf(fmt, args);
                 va_end(args);
               }));
#endif // _CCCL_HOSTED()
}

//! Logs the message when called from host code and logging is enabled
_CCCL_ATTRIBUTE_FORMAT(__printf__, 1, 2)
_CCCL_HOST_DEVICE_API inline void log([[maybe_unused]] const char* fmt, ...) noexcept
{
#if _CCCL_HOSTED() && !defined(CCCL_DISABLE_LOGGING)
  NV_IF_TARGET(NV_IS_HOST, ({
                 if (logging_enabled())
                 {
                   ::std::va_list args;
                   va_start(args, fmt);
                   ::vprintf(fmt, args);
                   va_end(args);
                 }
               }));

#endif // _CCCL_HOSTED()
}
} // namespace detail
CUB_NAMESPACE_END
