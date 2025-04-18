// -*- C++ -*-
//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===---------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FWD_STRING_VIEW_H
#define _LIBCUDACXX___FWD_STRING_VIEW_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__fwd/string.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _CharT, class _Traits = char_traits<_CharT>>
class _CCCL_TYPE_VISIBILITY_DEFAULT basic_string_view;

using string_view = basic_string_view<char>;
#if _LIBCUDACXX_HAS_CHAR8_T()
using u8string_view = basic_string_view<char8_t>;
#endif // _LIBCUDACXX_HAS_CHAR8_T()
using u16string_view = basic_string_view<char16_t>;
using u32string_view = basic_string_view<char32_t>;
using wstring_view   = basic_string_view<wchar_t>;

// clang-format off
template <class _CharT, class _Traits>
class _LIBCUDACXX_PREFERRED_NAME(string_view)
      _LIBCUDACXX_PREFERRED_NAME(wstring_view)
#if _LIBCUDACXX_HAS_CHAR8_T()
      _LIBCUDACXX_PREFERRED_NAME(u8string_view)
#endif // _LIBCUDACXX_HAS_CHAR8_T()
      _LIBCUDACXX_PREFERRED_NAME(u16string_view)
      _LIBCUDACXX_PREFERRED_NAME(u32string_view)
      basic_string_view;
// clang-format on
_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___FWD_STRING_VIEW_H
