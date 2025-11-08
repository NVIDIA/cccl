//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___CHARCONV_TO_CHARS_RESULT_H
#define _CUDA_STD___CHARCONV_TO_CHARS_RESULT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__system_error/errc.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

struct _CCCL_TYPE_VISIBILITY_DEFAULT to_chars_result
{
  char* ptr;
  errc ec;

  _CCCL_API constexpr explicit operator bool() const noexcept
  {
    return ec == errc{};
  }

  [[nodiscard]] _CCCL_API friend constexpr bool
  operator==(const to_chars_result& __lhs, const to_chars_result& __rhs) noexcept
  {
    return __lhs.ptr == __rhs.ptr && __lhs.ec == __rhs.ec;
  }

  [[nodiscard]] _CCCL_API friend constexpr bool
  operator!=(const to_chars_result& __lhs, const to_chars_result& __rhs) noexcept
  {
    return !(__lhs == __rhs);
  }
};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CHARCONV_TO_CHARS_RESULT_H
