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

#ifndef __cuda_std__
#  include <__config>
#endif // __cuda_std__

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/detail/libcxx/include/__fwd/string.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _CharT, class _Traits = char_traits<_CharT> >
class _LIBCUDACXX_TEMPLATE_VIS basic_string_view;

typedef basic_string_view<char> string_view;
#ifndef _LIBCUDACXX_NO_HAS_CHAR8_T
typedef basic_string_view<char8_t> u8string_view;
#endif
typedef basic_string_view<char16_t> u16string_view;
typedef basic_string_view<char32_t> u32string_view;
#ifndef _LIBCUDACXX_HAS_NO_WIDE_CHARACTERS
typedef basic_string_view<wchar_t> wstring_view;
#endif

// clang-format off
template <class _CharT, class _Traits>
class _LIBCUDACXX_PREFERRED_NAME(string_view)
#ifndef _LIBCUDACXX_HAS_NO_WIDE_CHARACTERS
      _LIBCUDACXX_PREFERRED_NAME(wstring_view)
#endif
#ifndef _LIBCUDACXX_NO_HAS_CHAR8_T
      _LIBCUDACXX_PREFERRED_NAME(u8string_view)
#endif
      _LIBCUDACXX_PREFERRED_NAME(u16string_view)
      _LIBCUDACXX_PREFERRED_NAME(u32string_view)
      basic_string_view;
// clang-format on
_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___FWD_STRING_VIEW_H
