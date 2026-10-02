//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___EXCEPTION_CUDA_ERROR_H
#define _CUDA_STD___EXCEPTION_CUDA_ERROR_H

// IWYU pragma: always_keep

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__driver/entry_point.h>
#include <cuda/std/__exception/exception_macros.h>
#include <cuda/std/__exception/msg_storage.h>
#include <cuda/std/__host_stdlib/cstdio>
#include <cuda/std/__host_stdlib/stdexcept>
#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/source_location>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

#if _CCCL_HAS_CTK()
using __cuda_error_t = ::cudaError_t;
#else
using __cuda_error_t = int;
#endif

#if _CCCL_HOSTED()
/**
 * @brief Describes a family of status codes that `cuda_error` can carry.
 *
 * The primary template is undefined. libcu++ specializes it for `cudaError_t` and `CUresult`. A library
 * whose API reports failures through its own status enumeration (cuBLAS, cuSOLVER, ...) specializes it
 * for that type, which makes `cuda_error` constructible from such a status and lets a handler recover
 * it exactly with `cuda_error::status<Status>()`. A specialization provides:
 *
 * @code
 * static constexpr unsigned id;                 // unique among domains; 0 and 1 belong to libcu++
 * static constexpr const char* name;            // e.g. "CUDA", "CUDA driver", "cuBLAS"
 * static const char* description(Status);      // human-readable text for a status, never null
 * @endcode
 */
template <class _Status>
struct cuda_status_domain;

namespace __detail
{
inline constexpr unsigned __cuda_runtime_domain = 0;
inline constexpr unsigned __cuda_driver_domain  = 1;
inline constexpr int __cuda_error_unknown       = 999; // ::cudaErrorUnknown, spelled out so the header needs no CTK

template <class _Status, class = void>
inline constexpr bool __is_cuda_status_v = false;
template <class _Status>
inline constexpr bool
  __is_cuda_status_v<_Status, ::cuda::std::void_t<decltype(::cuda::cuda_status_domain<_Status>::id)>> = true;

[[nodiscard]] _CCCL_HOST_API inline char* __format_cuda_error(
  ::cuda::__msg_storage& __msg_buffer,
  const ::cuda::std::source_location& __loc,
  const char* __api,
  const char* __error_str,
  const int __status,
  const char* __msg) noexcept
{
  ::snprintf(
    __msg_buffer.__buffer,
    512,
    "%s:%d %s%s%s(%d): %s",
    __loc.file_name(),
    __loc.line(),
    __api ? __api : "",
    __api ? " " : "",
    (__error_str != nullptr) ? __error_str : "cudaError",
    __status,
    __msg);
  return __msg_buffer.__buffer;
}
} // namespace __detail

#  if _CCCL_HAS_CTK()
template <>
struct cuda_status_domain<::cudaError_t>
{
  static constexpr unsigned id      = __detail::__cuda_runtime_domain;
  static constexpr const char* name = "CUDA";
  [[nodiscard]] _CCCL_HOST_API static const char* description(const ::cudaError_t __status)
  {
    return ::cuda::__driver::__getErrorString(__status);
  }
};

template <>
struct cuda_status_domain<::CUresult>
{
  static constexpr unsigned id      = __detail::__cuda_driver_domain;
  static constexpr const char* name = "CUDA driver";
  [[nodiscard]] _CCCL_HOST_API static const char* description(const ::CUresult __status)
  {
    return ::cuda::__driver::__getErrorString(__status);
  }
};
#  endif // _CCCL_HAS_CTK()

/**
 * @brief Exception thrown when a CUDA error is encountered.
 *
 * The exception carries the failing status as reported by the API that produced it, together with
 * the domain that status belongs to (CUDA Runtime, CUDA driver, or a library domain registered
 * through @ref cuda_status_domain) and the source location of the failure. `status()` keeps its
 * historical meaning, the status seen as a CUDA Runtime error code; `status<Status>()` recovers the
 * exact value.
 */
class cuda_error : public ::std::runtime_error
{
  int __raw_status_;
  unsigned __domain_;
  const char* __domain_name_;
  ::cuda::std::source_location __loc_;

  _CCCL_HOST_API cuda_error(
    const int __raw_status,
    const unsigned __domain,
    const char* __domain_name,
    const char* __error_str,
    const char* __msg,
    const char* __api,
    const ::cuda::std::source_location& __loc,
    __msg_storage __msg_buffer = {})
      : ::std::runtime_error(
          ::cuda::__detail::__format_cuda_error(__msg_buffer, __loc, __api, __error_str, __raw_status, __msg))
      , __raw_status_(__raw_status)
      , __domain_(__domain)
      , __domain_name_(__domain_name)
      , __loc_(__loc)
  {}

  // A runtime status with a caller-supplied description: `__throw_cuda_error<_Error>` uses it where the
  // driver may not be loadable yet.
  _CCCL_HOST_API cuda_error(
    const __cuda_error_t __status,
    const char* __error_str,
    const char* __msg,
    const char* __api,
    const ::cuda::std::source_location& __loc)
      : cuda_error{static_cast<int>(__status),
                   __detail::__cuda_runtime_domain,
                   "CUDA",
                   __error_str,
                   __msg,
                   __api,
                   __loc}
  {}

public:
  //! @brief Constructs from a CUDA Runtime status.
  _CCCL_HOST_API cuda_error(const __cuda_error_t __status,
                            const char* __msg,
                            const char* __api                         = nullptr,
                            const ::cuda::std::source_location& __loc = ::cuda::std::source_location::current())
      : cuda_error{__status,
#  if _CCCL_HAS_CTK()
                   ::cuda::__driver::__getErrorString(static_cast<::cudaError_t>(__status)),
#  else // ^^^ _CCCL_HAS_CTK() ^^^ / vvv !_CCCL_HAS_CTK() vvv
                   "cudaError",
#  endif // ^^^ !_CCCL_HAS_CTK() ^^^
                   __msg,
                   __api,
                   __loc}
  {}

  //! @brief Constructs from a status of another domain: `CUresult`, or any type with a
  //! @ref cuda_status_domain specialization.
  template <class _Status,
            ::cuda::std::enable_if_t<__detail::__is_cuda_status_v<_Status>
                                       && !::cuda::std::is_same_v<_Status, __cuda_error_t>,
                                     int> = 0>
  _CCCL_HOST_API cuda_error(const _Status __status,
                            const char* __msg,
                            const char* __api                         = nullptr,
                            const ::cuda::std::source_location& __loc = ::cuda::std::source_location::current())
      : cuda_error{static_cast<int>(__status),
                   cuda_status_domain<_Status>::id,
                   cuda_status_domain<_Status>::name,
                   cuda_status_domain<_Status>::description(__status),
                   __msg,
                   __api,
                   __loc}
  {}

  //! @brief The status seen as a CUDA Runtime error code. Exact for a runtime status. A driver status is
  //! converted numerically, which is how the runtime itself reports driver-originated failures. A status
  //! of any other domain reports `cudaErrorUnknown`; use `status<Status>()` for the exact value.
  [[nodiscard]] _CCCL_HOST_API constexpr auto status() const noexcept -> __cuda_error_t
  {
    return static_cast<__cuda_error_t>(
      __domain_ <= __detail::__cuda_driver_domain ? __raw_status_ : __detail::__cuda_error_unknown);
  }

  //! @brief Whether the stored status belongs to the domain of `_Status`.
  template <class _Status>
  [[nodiscard]] _CCCL_HOST_API constexpr bool holds() const noexcept
  {
    return __domain_ == cuda_status_domain<_Status>::id;
  }

  //! @brief The stored status as its own type. Precondition: `holds<_Status>()`.
  template <class _Status>
  [[nodiscard]] _CCCL_HOST_API _Status status() const noexcept
  {
    _CCCL_VERIFY(holds<_Status>(), "cuda_error::status<Status>(): the stored status belongs to another domain");
    return static_cast<_Status>(__raw_status_);
  }

  //! @brief The stored status as the API reported it, without interpretation.
  [[nodiscard]] _CCCL_HOST_API constexpr int raw_status() const noexcept
  {
    return __raw_status_;
  }

  //! @brief The domain of the stored status, `cuda_status_domain<Status>::id`.
  [[nodiscard]] _CCCL_HOST_API constexpr unsigned domain() const noexcept
  {
    return __domain_;
  }

  //! @brief The name of the stored status's domain, e.g. "CUDA driver".
  [[nodiscard]] _CCCL_HOST_API constexpr const char* domain_name() const noexcept
  {
    return __domain_name_;
  }

  //! @brief Where the error was raised.
  [[nodiscard]] _CCCL_HOST_API constexpr const ::cuda::std::source_location& location() const noexcept
  {
    return __loc_;
  }

  template <int _Error>
  [[noreturn]] friend _CCCL_HOST_API void
  __throw_cuda_error(const char* __msg, const char* __api, const ::cuda::std::source_location& __loc)
  {
    [[maybe_unused]] const char* __error_str{};
    if constexpr (_Error == /*::cudaErrorInvalidValue*/ 1)
    {
      __error_str = "invalid value";
    }
    else if constexpr (_Error == /*::cudaErrorInitializationError*/ 3)
    {
      __error_str = "initialization error";
    }
    else if constexpr (_Error == /*::cudaErrorNotSupported*/ 801)
    {
      __error_str = "operation not supported";
    }
    else if constexpr (_Error == /*::cudaErrorUnknown*/ 999)
    {
      __error_str = "unknown error";
    }
    else
    {
      static_assert(::cuda::std::__always_false_v<decltype(_Error)>, "unknown _Error");
    }
    _CCCL_THROW(::cuda::cuda_error, static_cast<__cuda_error_t>(_Error), __error_str, __msg, __api, __loc);
  }

  [[noreturn]] friend _CCCL_HOST_API void
  __throw_cuda_error(int __error, const char* __msg, const char* __api, const ::cuda::std::source_location& __loc)
  {
    _CCCL_THROW(::cuda::cuda_error, static_cast<__cuda_error_t>(__error), __msg, __api, __loc);
  }
};
#endif // _CCCL_HOSTED()

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___EXCEPTION_CUDA_ERROR_H
