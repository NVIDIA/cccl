// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#include <cub/util_debug.cuh> // for _CubLog

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__host_stdlib/cstdarg>
#include <cuda/std/__host_stdlib/cstdio>
#include <cuda/std/__host_stdlib/sstream>
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
                 _CCCL_DIAG_PUSH
                 _CCCL_DIAG_SUPPRESS_MSVC(4996) // 'getenv': This function or variable may be unsafe.
                 static const bool enabled = [] {
                   const char* const env = ::std::getenv("CCCL_EXPERIMENTAL_LOGGING");
                   return env != nullptr && ::std::atoi(env) != 0;
                 }();
                 _CCCL_DIAG_POP
                 return enabled;
               }),
               ({ return false; }));
#else // _CCCL_HOSTED() && !defined(CCCL_DISABLE_LOGGING)
  return false;
#endif // _CCCL_HOSTED() && !defined(CCCL_DISABLE_LOGGING)
}

//! Logs the message when called from host code, independently of whether logging is enabled via the
//! CCCL_EXPERIMENTAL_LOGGING env variable
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
#endif // _CCCL_HOSTED() && !defined(CCCL_DISABLE_LOGGING)
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

#endif // _CCCL_HOSTED() && !defined(CCCL_DISABLE_LOGGING)
}

template <typename Policy>
_CCCL_HOST_DEVICE_API void log_dispatch([[maybe_unused]] const char* device_alg,
                                        [[maybe_unused]] ::cuda::compute_capability cc,
                                        [[maybe_unused]] const Policy& active_policy) noexcept
{
#if _CCCL_HOSTED() && !defined(CCCL_DISABLE_LOGGING)
  NV_IF_TARGET(NV_IS_HOST, ({
                 if (logging_enabled())
                 {
                   ::std::stringstream ss;
                   ss << active_policy;
                   log_always("Dispatching %s on compute capability %d.%d with tuning: %s\n",
                              device_alg,
                              cc.major_cap(),
                              cc.minor_cap(),
                              ss.str().c_str());
                 }
               }))
#endif // _CCCL_HOSTED() && !defined(CCCL_DISABLE_LOGGING)
}
} // namespace detail
CUB_NAMESPACE_END

//! Logs a kernel launch (1D grid only). `kernel_name` must be a string literal. `fmt` is a string literal suffix
//! (may be `""`) appended to the standard "Invoking <kernel><<<grid, block, smem, stream>>>()" message, followed by
//! its corresponding printf-style args, e.g.:
//! `_CUB_LOG_KERNEL_LAUNCH("foo_kernel", grid_dim, block_dim, 0, stream, ", current bit: %d", current_bit);`
//!
//! Two logging mechanisms are supported:
//! - If `CUB_DEBUG_LOG` is defined, always prints via `_CubLog` (works from host and device code, e.g. under CDP).
//! - Otherwise, prints via `cub::detail::log` when logging is enabled via the CCCL_EXPERIMENTAL_LOGGING env
//!   variable (host code only).
#ifdef CUB_DEBUG_LOG
// TODO(bgruber): Remove along with _CubLog in CCCL 4.0
#  define _CUB_LOG_KERNEL_LAUNCH(kernel_name, grid_dim, block_dim, smem_bytes, stream, fmt, ...) \
    _CubLog("Invoking " kernel_name "<<<%d, %d, %zu, %lld>>>()" fmt "\n",                        \
            grid_dim,                                                                            \
            block_dim,                                                                           \
            static_cast<size_t>(smem_bytes),                                                     \
            reinterpret_cast<long long>(stream),                                                 \
            ##__VA_ARGS__)
#else // ^^^ CUB_DEBUG_LOG ^^^ / vvv !CUB_DEBUG_LOG vvv
#  define _CUB_LOG_KERNEL_LAUNCH(kernel_name, grid_dim, block_dim, smem_bytes, stream, fmt, ...) \
    CUB_NS_QUALIFIER::detail::log(                                                               \
      "Invoking " kernel_name "<<<%d, %d, %zu, %lld>>>()" fmt "\n",                              \
      grid_dim,                                                                                  \
      block_dim,                                                                                 \
      static_cast<size_t>(smem_bytes),                                                           \
      reinterpret_cast<long long>(stream),                                                       \
      ##__VA_ARGS__)
#endif // !CUB_DEBUG_LOG

//! Same as `_CUB_LOG_KERNEL_LAUNCH`, but for kernels launched with a 3D grid.
#ifdef CUB_DEBUG_LOG
// TODO(bgruber): Remove along with _CubLog in CCCL 4.0
#  define _CUB_LOG_KERNEL_LAUNCH_3D(                                                          \
    kernel_name, grid_dim_x, grid_dim_y, grid_dim_z, block_dim, smem_bytes, stream, fmt, ...) \
    _CubLog("Invoking " kernel_name "<<<{%d, %d, %d}, %d, %zu, %lld>>>()" fmt "\n",           \
            grid_dim_x,                                                                       \
            grid_dim_y,                                                                       \
            grid_dim_z,                                                                       \
            block_dim,                                                                        \
            static_cast<size_t>(smem_bytes),                                                  \
            reinterpret_cast<long long>(stream),                                              \
            ##__VA_ARGS__)
#else // ^^^ CUB_DEBUG_LOG ^^^ / vvv !CUB_DEBUG_LOG vvv
#  define _CUB_LOG_KERNEL_LAUNCH_3D(                                                          \
    kernel_name, grid_dim_x, grid_dim_y, grid_dim_z, block_dim, smem_bytes, stream, fmt, ...) \
    CUB_NS_QUALIFIER::detail::log(                                                            \
      "Invoking " kernel_name "<<<{%d, %d, %d}, %d, %zu, %lld>>>()" fmt "\n",                 \
      grid_dim_x,                                                                             \
      grid_dim_y,                                                                             \
      grid_dim_z,                                                                             \
      block_dim,                                                                              \
      static_cast<size_t>(smem_bytes),                                                        \
      reinterpret_cast<long long>(stream),                                                    \
      ##__VA_ARGS__)
#endif // !CUB_DEBUG_LOG
