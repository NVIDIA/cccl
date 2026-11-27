// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#include <cub/util_debug.cuh> // IWYU pragma: keep

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__device/compute_capability.h>
#include <cuda/std/__host_stdlib/cstdarg>
#include <cuda/std/__host_stdlib/cstdio> // IWYU pragma: keep
#include <cuda/std/__host_stdlib/iosfwd>
#include <cuda/std/cstdlib>

#include <nv/target>

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
                 if (_CUB::detail::logging_enabled())
                 {
                   ::std::va_list args;
                   va_start(args, fmt);
                   ::vprintf(fmt, args);
                   va_end(args);
                 }
               }));

#endif // _CCCL_HOSTED() && !defined(CCCL_DISABLE_LOGGING)
}

#if defined(CUB_DEBUG_LOG) || (_CCCL_HOSTED() && !defined(CCCL_DISABLE_LOGGING))
//! Emits the dispatch log line with the already formatted tuning description
_CCCL_HOST_DEVICE_API inline void
log_dispatch_message(const char* device_alg, ::cuda::compute_capability cc, const char* tuning) noexcept
{
#  ifdef CUB_DEBUG_LOG
  _CubLog(
    "Dispatching %s on compute capability %d.%d with tuning: %s\n", device_alg, cc.major_cap(), cc.minor_cap(), tuning);
#  else // ^^^ CUB_DEBUG_LOG ^^^ / vvv !CUB_DEBUG_LOG vvv
  _CUB::detail::log_always(
    "Dispatching %s on compute capability %d.%d with tuning: %s\n", device_alg, cc.major_cap(), cc.minor_cap(), tuning);
#  endif // !CUB_DEBUG_LOG
}

#  if _CCCL_HOSTED()
// CUB only relies on <iosfwd>, so ::std::basic_stringstream may be incomplete when this header is included. The
// preferred overload below is only viable when the stringstream is complete at the point of instantiation (i.e. the
// user included <sstream>). CharT is a template parameter so the check is dependent and deferred to instantiation.
// The overload requires _CCCL_HOSTED(): otherwise (e.g. NVRTC) the iosfwd wrapper declares nothing, and the template
// name ::std::basic_stringstream is already looked up when the template is defined.
template <typename Policy, typename CharT = char>
_CCCL_HOST_DEVICE_API auto
log_dispatch_impl(const char* device_alg, ::cuda::compute_capability cc, const Policy& active_policy, int) noexcept
  -> decltype(void(sizeof(::std::basic_stringstream<CharT>)))
{
  NV_IF_TARGET(NV_IS_HOST, ({
                 ::std::basic_stringstream<CharT> ss;
                 ss << active_policy;
                 _CUB::detail::log_dispatch_message(device_alg, cc, ss.str().c_str());
               }))
}
#  endif // _CCCL_HOSTED()

// Fallback when <sstream> was not included by the user: still log the dispatch, but without the tuning details
template <typename Policy>
_CCCL_HOST_DEVICE_API void
log_dispatch_impl(const char* device_alg, ::cuda::compute_capability cc, const Policy&, long) noexcept
{
  _CUB::detail::log_dispatch_message(device_alg, cc, "<unavailable, include <sstream> to log the tuning policy>");
}
#endif // CUB_DEBUG_LOG || (_CCCL_HOSTED() && !CCCL_DISABLE_LOGGING)

//! Logs the dispatch of a device algorithm together with the active tuning policy. Printing the policy requires the
//! user to include <sstream> (CUB itself only includes <iosfwd>); otherwise the tuning is reported as unavailable.
template <typename Policy>
_CCCL_HOST_DEVICE_API void log_dispatch([[maybe_unused]] const char* device_alg,
                                        [[maybe_unused]] ::cuda::compute_capability cc,
                                        [[maybe_unused]] const Policy& active_policy) noexcept
{
#ifdef CUB_DEBUG_LOG
  NV_IF_TARGET(NV_IS_HOST, ({ _CUB::detail::log_dispatch_impl(device_alg, cc, active_policy, 0); }))
#elif _CCCL_HOSTED() && !defined(CCCL_DISABLE_LOGGING)
  NV_IF_TARGET(NV_IS_HOST, ({
                 if (_CUB::detail::logging_enabled())
                 {
                   _CUB::detail::log_dispatch_impl(device_alg, cc, active_policy, 0);
                 }
               }))
#endif // !CUB_DEBUG_LOG && _CCCL_HOSTED() && !defined(CCCL_DISABLE_LOGGING)
}
} // namespace detail
CUB_NAMESPACE_END

#ifdef CUB_DEBUG_LOG
#  define _CUB_LOG_KERNEL_LAUNCH(                                                             \
    kernel_name, grid_dim_x, grid_dim_y, grid_dim_z, block_dim, smem_bytes, stream, fmt, ...) \
    _CubLog("Invoking " kernel_name "<<<{%u, %u, %u}, %u, %zu, %lld>>>()" fmt "\n",           \
            static_cast<unsigned int>(grid_dim_x),                                            \
            static_cast<unsigned int>(grid_dim_y),                                            \
            static_cast<unsigned int>(grid_dim_z),                                            \
            static_cast<unsigned int>(block_dim),                                             \
            static_cast<size_t>(smem_bytes),                                                  \
            reinterpret_cast<long long>(stream),                                              \
            ##__VA_ARGS__)
#else // ^^^ CUB_DEBUG_LOG ^^^ / vvv !CUB_DEBUG_LOG vvv
#  define _CUB_LOG_KERNEL_LAUNCH(                                                             \
    kernel_name, grid_dim_x, grid_dim_y, grid_dim_z, block_dim, smem_bytes, stream, fmt, ...) \
    _CUB::detail::log(                                                                        \
      "Invoking " kernel_name "<<<{%u, %u, %u}, %u, %zu, %lld>>>()" fmt "\n",                 \
      static_cast<unsigned int>(grid_dim_x),                                                  \
      static_cast<unsigned int>(grid_dim_y),                                                  \
      static_cast<unsigned int>(grid_dim_z),                                                  \
      static_cast<unsigned int>(block_dim),                                                   \
      static_cast<size_t>(smem_bytes),                                                        \
      reinterpret_cast<long long>(stream),                                                    \
      ##__VA_ARGS__)
#endif // !CUB_DEBUG_LOG
