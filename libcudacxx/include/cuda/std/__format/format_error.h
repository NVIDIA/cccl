//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___FORMAT_FORMAT_ERROR_H
#define _CUDA_STD___FORMAT_FORMAT_ERROR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__exception/terminate.h>

#if !_CCCL_COMPILER(NVRTC)
#  if __cpp_lib_format >= 201907L
#    include <format>
#  else // ^^^ __cpp_lib_format >= 201907L ^^^ / vvv __cpp_lib_format < 201907L vvv
#    include <stdexcept>
#  endif // ^^^ __cpp_lib_format < 201907L ^^^
#endif // !_CCCL_COMPILER(NVRTC)

#include <cuda/std/__cccl/prologue.h>

#if !_CCCL_COMPILER(NVRTC)

_CCCL_BEGIN_NAMESPACE_CUDA_STD_NOVERSION

#  if __cpp_lib_format >= 201907L
using ::std::format_error;
#  else // ^^^ __cpp_lib_format >= 201907L ^^^ / vvv __cpp_lib_format < 201907L vvv
class _CCCL_TYPE_VISIBILITY_DEFAULT format_error : public ::std::runtime_error
{
public:
  _CCCL_HOST_API explicit format_error(const ::std::string& __s)
      : ::std::runtime_error(__s)
  {}
  _CCCL_HOST_API explicit format_error(const char* __s)
      : ::std::runtime_error(__s)
  {}
  _CCCL_HIDE_FROM_ABI format_error(const format_error&)            = default;
  _CCCL_HIDE_FROM_ABI format_error& operator=(const format_error&) = default;
  _CCCL_HIDE_FROM_ABI virtual ~format_error() noexcept override    = default;
};
#  endif // ^^^ __cpp_lib_format < 201907L ^^^

_CCCL_END_NAMESPACE_CUDA_STD_NOVERSION

#endif // !_CCCL_COMPILER(NVRTC)

_CCCL_BEGIN_NAMESPACE_CUDA_STD

[[noreturn]] _CCCL_API inline void __throw_format_error(const char* __s)
{
#if _CCCL_HAS_EXCEPTIONS()
  NV_IF_ELSE_TARGET(NV_IS_HOST, (throw ::cuda::std::format_error(__s);), (::cuda::std::terminate();))
#else // ^^^ _CCCL_HAS_EXCEPTIONS() ^^^ / vvv !_CCCL_HAS_EXCEPTIONS() vvv
  ::cuda::std::terminate();
#endif // ^^^ !_CCCL_HAS_EXCEPTIONS() ^^^
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FORMAT_FORMAT_ERROR_H
