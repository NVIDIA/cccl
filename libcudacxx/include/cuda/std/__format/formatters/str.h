//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD__FORMAT_FORMATERS_STR_H
#define _CUDA_STD__FORMAT_FORMATERS_STR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__format/format_spec_parser.h>
#include <cuda/std/__format/formatter.h>
#include <cuda/std/__format/output_utils.h>
#include <cuda/std/__string/char_traits.h>
#include <cuda/std/__type_traits/is_null_pointer.h>
#include <cuda/std/__type_traits/is_pointer.h>
#include <cuda/std/__utility/to_underlying.h>
#include <cuda/std/string_view>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

//!
//! @brief Formatter for string types.
//!
//! @tparam _CharT The character type used for formatting.
//!
template <class _CharT>
struct __fmt_formatter_str
{
  //!
  //! @brief Parses the formatting specifications for string types.
  //!
  //! @param __ctx The parsing context containing the format specification.
  //! @return An iterator pointing to the end of the parsed format specification.
  //!
  template <class _ParseCtx>
  _CCCL_API constexpr typename _ParseCtx::iterator parse(_ParseCtx& __ctx)
  {
    typename _ParseCtx::iterator __result = __parser_.__parse(__ctx, ::cuda::std::__fmt_spec_fields_str());
    ::cuda::std::__fmt_process_display_type_str(__parser_.__type_);
    return __result;
  }

  //!
  //! @brief Formats a string value according to the parsed specifications.
  //!
  //! @param __value The string value to format.
  //! @param __ctx The formatting context where the formatted output will be stored.
  //! @return An iterator pointing to the end of the formatted output.
  //!
  template <class _Tp, class _FmtCtx>
  _CCCL_API typename _FmtCtx::iterator format(_Tp __value, _FmtCtx& __ctx) const
  {
    return __format(__value, __ctx);
  }

private:
  //!
  //! @brief Creates a parser for string formatting specifications.
  //!
  [[nodiscard]] _CCCL_API static constexpr __fmt_spec_parser<_CharT> __make_parser()
  {
    __fmt_spec_parser<_CharT> __parser{};
    __parser.__alignment_ = ::cuda::std::to_underlying(__fmt_spec_alignment::__left);
    return __parser;
  }

  //!
  //! @brief Formats a C-string according to the parsed specifications.
  //!
  //! @param __str The C-string to format.
  //! @param __ctx The formatting context where the formatted output will be stored.
  //! @return An iterator pointing to the end of the formatted output.
  //!
  template <class _FmtCtx>
  [[nodiscard]] _CCCL_API typename _FmtCtx::iterator __format(const _CharT* __str, _FmtCtx& __ctx) const
  {
    return __format(basic_string_view{__str}, __ctx);
  }

  //!
  //! @brief Formats a fixed-size array of characters according to the parsed specifications.
  //!
  //! @param __str The fixed-size array of characters to format.
  //! @param __ctx The formatting context where the formatted output will be stored.
  //! @return An iterator pointing to the end of the formatted output.
  //!
  template <class _FmtCtx, size_t _Size>
  [[nodiscard]] _CCCL_API typename _FmtCtx::iterator __format(const _CharT (&__str)[_Size], _FmtCtx& __ctx) const
  {
    const _CharT* const __pzero = char_traits<_CharT>::find(__str, _Size, _CharT{});
    _CCCL_ASSERT(__pzero != nullptr, "formatting a non-null-terminated array");
    return __format(basic_string_view{__str, static_cast<size_t>(__pzero - __str)}, __ctx);
  }

  //!
  //! @brief Formats a `basic_string_view` according to the parsed specifications.
  //!
  //! @param __str The `basic_string_view` to format.
  //! @param __ctx The formatting context where the formatted output will be stored.
  //! @return An iterator pointing to the end of the formatted output.
  //!
  template <class _FmtCtx, class _Traits>
  [[nodiscard]] _CCCL_API typename _FmtCtx::iterator
  __format(basic_string_view<_CharT, _Traits> __str, _FmtCtx& __ctx) const
  {
    basic_string_view __str2{__str.data(), __str.size()};
    return ::cuda::std::__fmt_write_string(__str2, __ctx.out(), __parser_.__get_parsed_std_spec(__ctx));
  }

  __fmt_spec_parser<_CharT> __parser_ = __make_parser(); //!< The parser for format specifications.
};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<const char*, char> : __fmt_formatter_str<char>
{};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<char*, char> : __fmt_formatter_str<char>
{};
template <size_t _Size>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<char[_Size], char> : __fmt_formatter_str<char>
{};
template <class _Traits>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<basic_string_view<char, _Traits>, char> : __fmt_formatter_str<char>
{};

#if _CCCL_HAS_WCHAR_T()
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<const char*, wchar_t> : __fmt_formatter_str<wchar_t>
{};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<char*, wchar_t> : __fmt_formatter_str<wchar_t>
{};
template <size_t _Size>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<char[_Size], wchar_t> : __fmt_formatter_str<wchar_t>
{};
template <class _Traits>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<basic_string_view<char, _Traits>, wchar_t> : __fmt_formatter_str<wchar_t>
{};

// Disable formatting string of type char to wchar_t output
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<char*, wchar_t> : __fmt_disabled_formatter
{};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<const char*, wchar_t> : __fmt_disabled_formatter
{};
template <size_t _Size>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<char[_Size], wchar_t> : __fmt_disabled_formatter
{};
template <class _Traits, class _Allocator>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
formatter<basic_string<char, _Traits, _Allocator>, wchar_t> : __fmt_disabled_formatter
{};
template <class _Traits>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<basic_string_view<char, _Traits>, wchar_t> : __fmt_disabled_formatter
{};
#endif // _CCCL_HAS_WCHAR_T()

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD__FORMAT_FORMATERS_STR_H
