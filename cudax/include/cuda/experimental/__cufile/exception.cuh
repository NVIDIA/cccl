//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__exception/cuda_error.h>
#include <cuda/std/__exception/terminate.h>
#include <cuda/std/source_location>

#include <cstdio>
#include <stdexcept>

#include <cufile.h>

namespace cuda::experimental
{
#if _CCCL_HAS_CTK()
using __cufile_error_t = ::CUfileOpError;
#else // ^^^ _CCCL_HAS_CTK() ^^^ // vvv !_CCCL_HAS_CTK() vvv
using __cufile_error_t = int;
#endif // ^^^ !_CCCL_HAS_CTK() ^^^

struct __cufile_msg_storage
{
  char __buffer[512]{};
};

static char* __format_cufile_error_message(
  __cufile_msg_storage& __msg_buffer,
  const __cufile_error_t __status,
  const char* __msg,
  const char* __api                  = nullptr,
  ::cuda::std::source_location __loc = ::cuda::std::source_location::current()) noexcept
{
  ::snprintf(
    __msg_buffer.__buffer,
    512,
    "%s:%d %s%s%s(%d): %s",
    __loc.file_name(),
    __loc.line(),
    __api ? __api : "",
    __api ? " " : "",
#if _CCCL_HAS_CTK()
    ::cufileop_status_error(::CUfileOpError{__status}),
#else // ^^^ _CCCL_HAS_CTK() ^^^ / vvv !_CCCL_HAS_CTK() vvv
    "cuFile error",
#endif // ^^^ !_CCCL_HAS_CTK() ^^^
    __status,
    __msg);
  return __msg_buffer.__buffer;
}

//! @brief Exception class for errors from cuFile APIs.
class cufile_error : public ::std::runtime_error
{
  __cufile_error_t __status_; //!< The cuFile error code.

public:
  _CCCL_HOST_API cufile_error(
    __cufile_error_t __status,
    const char* __msg,
    const char* __api,
    ::cuda::std::source_location loc  = ::cuda::std::source_location::current(),
    __cufile_msg_storage __msg_buffer = {})
      : ::std::runtime_error{__format_cufile_error_message(__msg_buffer, __status, __msg, __api, loc)}
      , __status_{__status}
  {}

  [[nodiscard]] _CCCL_HOST_API __cufile_error_t status() const noexcept
  {
    return __status_;
  }
};

[[noreturn]] _CCCL_HOST_API inline void __throw_cufile_error(
  __cufile_error_t __status,
  const char* __msg,
  const char* __api,
  ::cuda::std::source_location __loc = ::cuda::std::source_location::current())
{
#if _CCCL_HAS_EXCEPTIONS()
  throw cufile_error{__status, __msg, __api, __loc};
#else // ^^^ _CCCL_CUDA_COMPILATION() ^^^ / vvv !_CCCL_CUDA_COMPILATION() vvv
  ::cuda::std::terminate();
#endif // !_CCCL_CUDA_COMPILATION()
}

//! @brief Macro to call a cuFile API and throw a cufile_error or cuda_error if it fails.
#define _CCCL_TRY_CUFILE_API(_NAME, _MSG, ...)                                                              \
  do                                                                                                        \
  {                                                                                                         \
    const ::CUfileError_t __cufile_error_status = _NAME(__VA_ARGS__);                                       \
    switch (__cufile_error_status.err)                                                                      \
    {                                                                                                       \
      case ::CU_FILE_SUCCESS:                                                                               \
        break;                                                                                              \
      case ::CU_FILE_CUDA_DRIVER_ERROR:                                                                     \
        ::cuda::__throw_cuda_error(static_cast<::cudaError_t>(__cufile_error_status.cu_err), _MSG, #_NAME); \
      default:                                                                                              \
        __throw_cufile_error(__cufile_error_status.err, _MSG, #_NAME);                                      \
    }                                                                                                       \
  } while (0)
} // namespace cuda::experimental
