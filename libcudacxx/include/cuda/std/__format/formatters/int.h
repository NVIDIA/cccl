//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD__FORMAT_FORMATERS_INT_H
#define _CUDA_STD__FORMAT_FORMATERS_INT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__format/format_integral.h>
#include <cuda/std/__format/format_spec_parser.h>
#include <cuda/std/__format/formatter.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/make_nbit_int.h>
#include <cuda/std/climits>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

//!
//! @brief Formatter for integer types.
//!
//! @tparam _CharT The character type used for formatting.
//!
template <class _CharT>
struct __fmt_formatter_int
{
  //!
  //! @brief Parses the formatting specifications for integer types.
  //!
  //! @param __ctx The parsing context containing the format specification.
  //! @return An iterator pointing to the end of the parsed format specification.
  //!
  template <class _ParseCtx>
  _CCCL_API constexpr typename _ParseCtx::iterator parse(_ParseCtx& __ctx)
  {
    typename _ParseCtx::iterator __result = __parser_.__parse(__ctx, ::cuda::std::__fmt_spec_fields_int());
    ::cuda::std::__fmt_process_parsed_int(__parser_);
    return __result;
  }

  //!
  //! @brief Formats an integer value according to the parsed specifications.
  //!
  //! @param __value The integer value to format.
  //! @param __ctx The formatting context where the formatted output will be stored.
  //! @return An iterator pointing to the end of the formatted output.
  //!
  template <class _Tp, class _FmtCtx>
  _CCCL_API typename _FmtCtx::iterator format(_Tp __value, _FmtCtx& __ctx) const
  {
    const auto __specs = __parser_.__get_parsed_std_spec(__ctx);

    if (__specs.__std_.__type_ == __fmt_spec_type::__char)
    {
      return ::cuda::std::__fmt_format_char(__value, __ctx.out(), __specs);
    }

    using _Type =
      __make_nbit_int_t<::cuda::std::max(sizeof(_Tp) * CHAR_BIT, sizeof(int32_t) * CHAR_BIT), is_signed_v<_Tp>>;

    // Reduce the number of instantiation of the integer formatter
    return ::cuda::std::__fmt_format_int(static_cast<_Type>(__value), __ctx, __specs);
  }

private:
  __fmt_spec_parser<_CharT> __parser_; //!< The parser for format specifications.
};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<signed char, char> : __fmt_formatter_int<char>
{};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<short, char> : __fmt_formatter_int<char>
{};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<int, char> : __fmt_formatter_int<char>
{};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<long, char> : __fmt_formatter_int<char>
{};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<long long, char> : __fmt_formatter_int<char>
{};
#if _CCCL_HAS_INT128()
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<__int128_t, char> : __fmt_formatter_int<char>
{};
#endif // _CCCL_HAS_INT128()
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<unsigned char, char> : __fmt_formatter_int<char>
{};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<unsigned short, char> : __fmt_formatter_int<char>
{};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<unsigned, char> : __fmt_formatter_int<char>
{};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<unsigned long, char> : __fmt_formatter_int<char>
{};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<unsigned long long, char> : __fmt_formatter_int<char>
{};
#if _CCCL_HAS_INT128()
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<__uint128_t, char> : __fmt_formatter_int<char>
{};
#endif // _CCCL_HAS_INT128()

#if _CCCL_HAS_WCHAR_T()
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<signed char, wchar_t> : __fmt_formatter_int<wchar_t>
{};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<short, wchar_t> : __fmt_formatter_int<wchar_t>
{};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<int, wchar_t> : __fmt_formatter_int<wchar_t>
{};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<long, wchar_t> : __fmt_formatter_int<wchar_t>
{};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<long long, wchar_t> : __fmt_formatter_int<wchar_t>
{};
#  if _CCCL_HAS_INT128()
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<__int128_t, wchar_t> : __fmt_formatter_int<wchar_t>
{};
#  endif // _CCCL_HAS_INT128()
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<unsigned char, wchar_t> : __fmt_formatter_int<wchar_t>
{};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<unsigned short, wchar_t> : __fmt_formatter_int<wchar_t>
{};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<unsigned, wchar_t> : __fmt_formatter_int<wchar_t>
{};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<unsigned long, wchar_t> : __fmt_formatter_int<wchar_t>
{};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<unsigned long long, wchar_t> : __fmt_formatter_int<wchar_t>
{};
#  if _CCCL_HAS_INT128()
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<__uint128_t, wchar_t> : __fmt_formatter_int<wchar_t>
{};
#  endif // _CCCL_HAS_INT128()
#endif // _CCCL_HAS_WCHAR_T()

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD__FORMAT_FORMATERS_INT_H
