// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___EXCEPTION_CUDA_ERROR_H
#define _LIBCUDACXX___EXCEPTION_CUDA_ERROR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CUDA_COMPILER()
#  include <cuda_runtime_api.h>
#endif // _CCCL_HAS_CUDA_COMPILER()

#include <cuda/std/__exception/terminate.h>
#include <cuda/std/source_location>

#if !_CCCL_COMPILER(NVRTC)
#  include <cstdio>
#  include <stdexcept>
#endif // !_CCCL_COMPILER(NVRTC)

#include <nv/target>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

#if _CCCL_HAS_EXCEPTIONS()

namespace __detail
{

struct __msg_storage
{
  char __buffer[512]{0};
};

static char* __format_cuda_error(
  __msg_storage& __msg_buffer,
  const int __status,
  const char* __msg,
  const char* __api                 = nullptr,
  _CUDA_VSTD::source_location __loc = _CUDA_VSTD::source_location::current()) noexcept
{
  ::snprintf(
    __msg_buffer.__buffer,
    512,
    "%s:%d %s%s%s(%d): %s",
    __loc.file_name(),
    __loc.line(),
    __api ? __api : "",
    __api ? " " : "",
#  if _CCCL_HAS_CUDA_COMPILER()
    ::cudaGetErrorString(::cudaError_t(__status)),
#  else
    "cudaError",
#  endif
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
public:
  cuda_error(const int __status,
             const char* __msg,
             const char* __api                    = nullptr,
             _CUDA_VSTD::source_location __loc    = _CUDA_VSTD::source_location::current(),
             __detail::__msg_storage __msg_buffer = {}) noexcept
      : ::std::runtime_error(__detail::__format_cuda_error(__msg_buffer, __status, __msg, __api, __loc))
  {}
};

[[noreturn]] _LIBCUDACXX_HIDE_FROM_ABI void __throw_cuda_error(
  [[maybe_unused]] const int __status,
  [[maybe_unused]] const char* __msg,
  [[maybe_unused]] const char* __api                 = nullptr,
  [[maybe_unused]] _CUDA_VSTD::source_location __loc = _CUDA_VSTD::source_location::current())
{
  NV_IF_ELSE_TARGET(NV_IS_HOST,
                    (throw ::cuda::cuda_error(__status, __msg, __api, __loc);), //
                    (_CUDA_VSTD_NOVERSION::terminate();))
}
#else // ^^^ _CCCL_HAS_EXCEPTIONS() ^^^ / vvv !_CCCL_HAS_EXCEPTIONS() vvv
class cuda_error
{
public:
  _LIBCUDACXX_HIDE_FROM_ABI cuda_error(
    const int,
    const char*,
    const char*                 = nullptr,
    _CUDA_VSTD::source_location = _CUDA_VSTD::source_location::current()) noexcept
  {}
};

[[noreturn]] _LIBCUDACXX_HIDE_FROM_ABI void __throw_cuda_error(
  const int, const char*, const char* = nullptr, _CUDA_VSTD::source_location = _CUDA_VSTD::source_location::current())
{
  _CUDA_VSTD_NOVERSION::terminate();
}
#endif // !_CCCL_HAS_EXCEPTIONS()

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _LIBCUDACXX___EXCEPTION_CUDA_ERROR_H
