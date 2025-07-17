//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FORMAT_FORMATTER_H
#define _LIBCUDACXX___FORMAT_FORMATTER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/string_view>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

struct __disabled_formatter
{
  __disabled_formatter()                                       = delete;
  __disabled_formatter(const __disabled_formatter&)            = delete;
  __disabled_formatter(__disabled_formatter&&)                 = delete;
  __disabled_formatter& operator=(const __disabled_formatter&) = delete;
  __disabled_formatter& operator=(__disabled_formatter&&)      = delete;
};

template <class _Tp>
struct __dummy_formatter
{
  template <class _ParseContext>
  _CCCL_API constexpr typename _ParseContext::iterator parse(_ParseContext& __ctx)
  {
    _CCCL_ASSERT(false, "unimplemented formatter parse method called");
    return __ctx.begin();
  }

  template <class _FormatContext>
  _CCCL_API typename _FormatContext::iterator format(const _Tp& __value, _FormatContext& __ctx) const
  {
    _CCCL_ASSERT(false, "unimplemented formatter format method called");
    (void) __value;
    return __ctx.out();
  }
};

/// The default formatter template.
///
/// [format.formatter.spec]/5
/// If F is a disabled specialization of formatter, these values are false:
/// - is_default_constructible_v<F>,
/// - is_copy_constructible_v<F>,
/// - is_move_constructible_v<F>,
/// - is_copy_assignable_v<F>, and
/// - is_move_assignable_v<F>.
template <class _Tp, class _CharT = char>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter : __disabled_formatter
{};

// todo: implement the formatters
template <>
struct formatter<bool, char> : __dummy_formatter<bool>
{};
template <>
struct formatter<char, char> : __dummy_formatter<char>
{};
template <>
struct formatter<float, char> : __dummy_formatter<float>
{};
template <>
struct formatter<double, char> : __dummy_formatter<double>
{};
#if _CCCL_HAS_LONG_DOUBLE()
template <>
struct formatter<long double, char> : __dummy_formatter<long double>
{};
#endif // _CCCL_HAS_LONG_DOUBLE()
template <>
struct formatter<signed char, char> : __dummy_formatter<signed char>
{};
template <>
struct formatter<short, char> : __dummy_formatter<short>
{};
template <>
struct formatter<int, char> : __dummy_formatter<int>
{};
template <>
struct formatter<long, char> : __dummy_formatter<long>
{};
template <>
struct formatter<long long, char> : __dummy_formatter<long long>
{};
#if _CCCL_HAS_INT128()
template <>
struct formatter<__int128_t, char> : __dummy_formatter<__int128_t>
{};
#endif // _CCCL_HAS_INT128()
template <>
struct formatter<unsigned char, char> : __dummy_formatter<unsigned char>
{};
template <>
struct formatter<unsigned short, char> : __dummy_formatter<unsigned short>
{};
template <>
struct formatter<unsigned, char> : __dummy_formatter<unsigned>
{};
template <>
struct formatter<unsigned long, char> : __dummy_formatter<unsigned long>
{};
template <>
struct formatter<unsigned long long, char> : __dummy_formatter<unsigned long long>
{};
#if _CCCL_HAS_INT128()
template <>
struct formatter<__uint128_t, char> : __dummy_formatter<__uint128_t>
{};
#endif // _CCCL_HAS_INT128()
template <>
struct formatter<nullptr_t, char> : __dummy_formatter<nullptr_t>
{};
template <>
struct formatter<void*, char> : __dummy_formatter<void*>
{};
template <>
struct formatter<const void*, char> : __dummy_formatter<const void*>
{};
template <>
struct formatter<const char*, char> : __dummy_formatter<const char*>
{};
template <>
struct formatter<char*, char> : __dummy_formatter<char*>
{};
template <size_t _Size>
struct formatter<char[_Size], char> : __dummy_formatter<char[_Size]>
{};
template <class _Traits>
struct formatter<basic_string_view<char, _Traits>, char> : __dummy_formatter<basic_string_view<char, _Traits>>
{};

#if _CCCL_HAS_WCHAR_T()
template <>
struct formatter<bool, wchar_t> : __dummy_formatter<bool>
{};
template <>
struct formatter<char, wchar_t> : __dummy_formatter<wchar_t>
{};
template <>
struct formatter<wchar_t, wchar_t> : __dummy_formatter<wchar_t>
{};
struct formatter<float, wchar_t> : __dummy_formatter<float>
{};
template <>
struct formatter<double, wchar_t> : __dummy_formatter<double>
{};
#  if _CCCL_HAS_LONG_DOUBLE()
template <>
struct formatter<long double, wchar_t> : __dummy_formatter<long double>
{};
#  endif // _CCCL_HAS_LONG_DOUBLE()
template <>
struct formatter<signed char, wchar_t> : __dummy_formatter<signed char>
{};
template <>
struct formatter<short, wchar_t> : __dummy_formatter<short>
{};
template <>
struct formatter<int, wchar_t> : __dummy_formatter<int>
{};
template <>
struct formatter<long, wchar_t> : __dummy_formatter<long>
{};
template <>
struct formatter<long long, wchar_t> : __dummy_formatter<long long>
{};
#  if _CCCL_HAS_INT128()
template <>
struct formatter<__int128_t, wchar_t> : __dummy_formatter<__int128_t>
{};
#  endif // _CCCL_HAS_INT128()
template <>
struct formatter<unsigned char, wchar_t> : __dummy_formatter<unsigned char>
{};
template <>
struct formatter<unsigned short, wchar_t> : __dummy_formatter<unsigned short>
{};
template <>
struct formatter<unsigned, wchar_t> : __dummy_formatter<unsigned>
{};
template <>
struct formatter<unsigned long, wchar_t> : __dummy_formatter<unsigned long>
{};
template <>
struct formatter<unsigned long long, wchar_t> : __dummy_formatter<unsigned long long>
{};
#  if _CCCL_HAS_INT128()
template <>
struct formatter<__uint128_t, wchar_t> : __dummy_formatter<__uint128_t>
{};
#  endif // _CCCL_HAS_INT128()
template <>
struct formatter<nullptr_t, wchar_t> : __dummy_formatter<nullptr_t>
{};
template <>
struct formatter<void*, wchar_t> : __dummy_formatter<void*>
{};
template <>
struct formatter<const void*, wchar_t> : __dummy_formatter<const void*>
{};
template <>
struct formatter<const wchar_t*, wchar_t> : __dummy_formatter<const wchar_t*>
{};
template <>
struct formatter<wchar_t*, wchar_t> : __dummy_formatter<wchar_t*>
{};
template <size_t _Size>
struct formatter<wchar_t[_Size], wchar_t> : __dummy_formatter<wchar_t[_Size]>
{};
template <class _Traits>
struct formatter<basic_string_view<wchar_t, _Traits>, wchar_t> : __dummy_formatter<basic_string_view<wchar_t, _Traits>>
{};
template <>
struct formatter<char*, wchar_t> : __disabled_formatter
{};
template <>
struct formatter<const char*, wchar_t> : __disabled_formatter
{};
template <size_t _Size>
struct formatter<char[_Size], wchar_t> : __disabled_formatter
{};
template <class _Traits>
struct formatter<basic_string_view<char, _Traits>, wchar_t> : __disabled_formatter
{};
#endif // _CCCL_HAS_WCHAR_T()

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___FORMAT_FORMATTER_H
