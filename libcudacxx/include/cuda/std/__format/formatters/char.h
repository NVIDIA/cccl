//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD__FORMAT_FORMATERS_CHAR_H
#define _CUDA_STD__FORMAT_FORMATERS_CHAR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__format/format_integral.h>
#include <cuda/std/__format/format_spec_parser.h>
#include <cuda/std/__format/formatter.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/make_unsigned.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

//!
//! @brief Formatter for character types.
//!
//! @tparam _CharT The character type used for formatting.
//!
template <class _CharT>
struct __fmt_formatter_char
{
  //!
  //! @brief Parses the formatting specifications for character types.
  //!
  //! @param __ctx The parsing context containing the format specification.
  //! @return An iterator pointing to the end of the parsed format specification.
  //!
  template <class _ParseCtx>
  _CCCL_API constexpr typename _ParseCtx::iterator parse(_ParseCtx& __ctx)
  {
    typename _ParseCtx::iterator __result = __parser_.__parse(__ctx, ::cuda::std::__fmt_spec_fields_int());
    ::cuda::std::__fmt_process_parsed_char(__parser_);
    return __result;
  }

  //!
  //! @brief Formats a character value according to the parsed specifications.
  //!
  //! @param __value The character value to format.
  //! @param __ctx The formatting context where the formatted output will be stored.
  //! @return An iterator pointing to the end of the formatted output.
  //!
  template <class _FmtCtx>
  _CCCL_API typename _FmtCtx::iterator format(_CharT __value, _FmtCtx& __ctx) const
  {
    using _Up = make_unsigned_t<_CharT>;

    const auto __specs = __parser_.__get_parsed_std_spec(__ctx);

    if (__parser_.__type_ == __fmt_spec_type::__default || __parser_.__type_ == __fmt_spec_type::__char)
    {
      return ::cuda::std::__fmt_format_char(__value, __ctx.out(), __specs);
    }

    if constexpr (sizeof(_CharT) <= sizeof(unsigned))
    {
      return ::cuda::std::__fmt_format_int(static_cast<unsigned>(static_cast<_Up>(__value)), __ctx, __specs);
    }
    else
    {
      return ::cuda::std::__fmt_format_int(static_cast<_Up>(__value), __ctx, __specs);
    }
  }

#if _CCCL_HAS_WCHAR_T()
  //!
  //! @brief Formats a character value as a wide character.
  //!
  //! @param __value The character value to format.
  //! @param __ctx The formatting context where the formatted output will be stored.
  //! @return An iterator pointing to the end of the formatted output.
  //!
  //! @details This overload allows formatting of a `char` type as a `wchar_t`.
  //!
  _CCCL_TEMPLATE(class _FmtCtx, class _CharT2 = _CharT)
  _CCCL_REQUIRES(is_same_v<_CharT2, wchar_t>)
  _CCCL_API typename _FmtCtx::iterator format(char __value, _FmtCtx& __ctx) const
  {
    return format(static_cast<wchar_t>(static_cast<unsigned char>(__value)), __ctx);
  }
#endif // _CCCL_HAS_WCHAR_T()

private:
  __fmt_spec_parser<_CharT> __parser_; //!< The parser for format specifications.
};

template <>
struct formatter<char, char> : __fmt_formatter_char<char>
{};

#if _CCCL_HAS_WCHAR_T()
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<char, wchar_t> : __fmt_formatter_char<wchar_t>
{};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<wchar_t, wchar_t> : __fmt_formatter_char<wchar_t>
{};
#endif // _CCCL_HAS_WCHAR_T()

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD__FORMAT_FORMATERS_CHAR_H
