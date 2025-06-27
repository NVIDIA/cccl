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

#if !_CCCL_COMPILER(NVRTC)

#  include <cuda/std/__exception/exception_macros.h>
#  include <cuda/std/__exception/terminate.h>
#  include <cuda/std/source_location>

#  include <nv/target>

#  include <cstdio>
#  include <stdexcept>

#  include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

#  if _CCCL_HAS_CTK()
using __cuda_error_t = ::cudaError_t;
#  else
using __cuda_error_t = int;
#  endif

namespace __detail
{

struct __msg_storage
{
  char __buffer[512]{0};
};

[[nodiscard]] _CCCL_HOST_API inline char* __format_cuda_error(
  __msg_storage& __msg_buffer,
  const __cuda_error_t __status,
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
#  if _CCCL_HAS_CTK()
    ::cudaGetErrorString(__status),
#  else // ^^^ _CCCL_HAS_CTK() ^^^ / vvv !_CCCL_HAS_CTK() vvv
    "cudaError",
#  endif // ^^^ !_CCCL_HAS_CTK() ^^^
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
  _CCCL_HOST_API cuda_error(
    const __cuda_error_t __status,
    const char* __msg,
    const char* __api                    = nullptr,
    _CUDA_VSTD::source_location __loc    = _CUDA_VSTD::source_location::current(),
    __detail::__msg_storage __msg_buffer = {}) noexcept
      : ::std::runtime_error(__detail::__format_cuda_error(__msg_buffer, __status, __msg, __api, __loc))
      , __status_(__status)
  {}

  [[nodiscard]] _CCCL_HOST_API auto status() const noexcept -> __cuda_error_t
  {
    return __status_;
  }

private:
  __cuda_error_t __status_;
};

[[noreturn]] _CCCL_API inline void __throw_cuda_error(
  [[maybe_unused]] const __cuda_error_t __status,
  [[maybe_unused]] const char* __msg,
  [[maybe_unused]] const char* __api                 = nullptr,
  [[maybe_unused]] _CUDA_VSTD::source_location __loc = _CUDA_VSTD::source_location::current())
{
#  if _CCCL_HAS_CTK()
  NV_IF_TARGET(NV_IS_HOST, (::cudaGetLastError();)) // clear CUDA error state
#  endif // _CCCL_HAS_CTK()
  _CCCL_THROW(::cuda::cuda_error(__status, __msg, __api, __loc));
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)

#endif // _LIBCUDACXX___EXCEPTION_CUDA_ERROR_H
