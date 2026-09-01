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
#include <cuda/std/source_location>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

#if _CCCL_HAS_CTK()
using __cuda_error_t = ::cudaError_t;
#else
using __cuda_error_t = int;
#endif

#if _CCCL_HOSTED()
namespace __detail
{
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

/**
 * @brief Exception thrown when a CUDA error is encountered.
 */
class cuda_error : public ::std::runtime_error
{
  __cuda_error_t __status_;

  _CCCL_HOST_API cuda_error(
    const __cuda_error_t __status,
    const char* __error_str,
    const char* __msg,
    const char* __api,
    const ::cuda::std::source_location& __loc,
    __msg_storage __msg_buffer = {})
      : ::std::runtime_error(
          ::cuda::__detail::__format_cuda_error(__msg_buffer, __loc, __api, __error_str, __status, __msg))
      , __status_(__status)
  {}

public:
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

  [[nodiscard]] _CCCL_HOST_API constexpr auto status() const noexcept -> __cuda_error_t
  {
    return __status_;
  }

  template <int _Error>
  [[noreturn]] friend _CCCL_HOST_API void
  __throw_cuda_error(const char* __msg, const char* __api, const ::cuda::std::source_location& __loc)
  {
    const char* __error_str{};
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
};
#endif // _CCCL_HOSTED()

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___EXCEPTION_CUDA_ERROR_H
