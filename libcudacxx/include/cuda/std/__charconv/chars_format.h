//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CHARCONV_CHARS_FORMAT_H
#define _LIBCUDACXX___CHARCONV_CHARS_FORMAT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__utility/to_underlying.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

enum class chars_format
{
  // We intentionally don't use `to_underlying(std::chars_format::XXX)` for XXX values because we want to avoid the risk
  // of the value mismatch between host's standard library and our definitions for NVRTC
  scientific = 0x1,
  fixed      = 0x2,
  hex        = 0x4,
  general    = fixed | scientific,
};

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr chars_format operator~(chars_format __v) noexcept
{
  return chars_format(~_CUDA_VSTD::to_underlying(__v));
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr chars_format operator&(chars_format __lhs, chars_format __rhs) noexcept
{
  return chars_format(_CUDA_VSTD::to_underlying(__lhs) & _CUDA_VSTD::to_underlying(__rhs));
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr chars_format operator|(chars_format __lhs, chars_format __rhs) noexcept
{
  return chars_format(_CUDA_VSTD::to_underlying(__lhs) | _CUDA_VSTD::to_underlying(__rhs));
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr chars_format operator^(chars_format __lhs, chars_format __rhs) noexcept
{
  return chars_format(_CUDA_VSTD::to_underlying(__lhs) ^ _CUDA_VSTD::to_underlying(__rhs));
}

_LIBCUDACXX_HIDE_FROM_ABI constexpr chars_format& operator&=(chars_format& __lhs, chars_format __rhs) noexcept
{
  __lhs = __lhs & __rhs;
  return __lhs;
}

_LIBCUDACXX_HIDE_FROM_ABI constexpr chars_format& operator|=(chars_format& __lhs, chars_format __rhs) noexcept
{
  __lhs = __lhs | __rhs;
  return __lhs;
}

_LIBCUDACXX_HIDE_FROM_ABI constexpr chars_format& operator^=(chars_format& __lhs, chars_format __rhs) noexcept
{
  __lhs = __lhs ^ __rhs;
  return __lhs;
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___CHARCONV_CHARS_FORMAT_H
