//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___FORMAT_DYNAMIC_FORMAT_H
#define _CUDA_STD___FORMAT_DYNAMIC_FORMAT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__fwd/format.h>
#include <cuda/std/string_view>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _CharT>
struct __dynamic_format_string
{
private:
  basic_string_view<_CharT> __str_;

  template <class _Cp, class... _Args>
  friend struct basic_format_string;

public:
  _CCCL_API __dynamic_format_string(basic_string_view<_CharT> __s) noexcept
      : __str_(__s)
  {}

  __dynamic_format_string(const __dynamic_format_string&)            = delete;
  __dynamic_format_string(__dynamic_format_string&&)                 = delete;
  __dynamic_format_string& operator=(const __dynamic_format_string&) = delete;
  __dynamic_format_string& operator=(__dynamic_format_string&&)      = delete;
};

[[nodiscard]] _CCCL_API inline __dynamic_format_string<char> dynamic_format(string_view __fmt) noexcept
{
  return __fmt;
}
#if _CCCL_HAS_WCHAR_T()
[[nodiscard]] _CCCL_API inline __dynamic_format_string<wchar_t> dynamic_format(wstring_view __fmt) noexcept
{
  return __fmt;
}
#endif // _CCCL_HAS_WCHAR_T()

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FORMAT_DYNAMIC_FORMAT_H
