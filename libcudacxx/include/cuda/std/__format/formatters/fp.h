//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD__FORMAT_FORMATERS_FP_H
#define _CUDA_STD__FORMAT_FORMATERS_FP_H

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

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

//!
//! @brief Formatter for floating-point types.
//!
//! @tparam _CharT The character type used for formatting.
//!
template <class _CharT>
struct __fmt_formatter_fp
{
  //!
  //! @brief Parses the formatting specifications for floating-point types.
  //!
  //! @param __ctx The parsing context containing the format specification.
  //! @return An iterator pointing to the end of the parsed format specification.
  //!
  template <class _ParseCtx>
  _CCCL_API constexpr typename _ParseCtx::iterator parse(_ParseCtx& __ctx)
  {
    typename _ParseCtx::iterator __result = __parser_.__parse(__ctx, ::cuda::std::__fmt_spec_fields_fp());
    ::cuda::std::__fmt_process_parsed_fp(__parser_);
    return __result;
  }

  //!
  //! @brief Formats a floating-point value according to the parsed specifications.
  //!
  //! @param __value The floating-point value to format.
  //! @param __ctx The formatting context where the formatted output will be stored.
  //! @return An iterator pointing to the end of the formatted output.
  //!
  template <class _Tp, class _FmtCtx>
  _CCCL_API typename _FmtCtx::iterator format(_Tp __value, _FmtCtx& __ctx) const
  {
    _CCCL_ASSERT(false, "formatter</*floating-point-type*/>::format() is not implemented");
    (void) __value;
    return __ctx.out();
  }

private:
  __fmt_spec_parser<_CharT> __parser_; //!< The parser for format specifications.
};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<float, char> : __fmt_formatter_fp<char>
{};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<double, char> : __fmt_formatter_fp<char>
{};
#if _CCCL_HAS_LONG_DOUBLE()
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<long double, char> : __fmt_formatter_fp<char>
{};
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _CCCL_HAS_WCHAR_T()
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<float, wchar_t> : __fmt_formatter_fp<wchar_t>
{};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<double, wchar_t> : __fmt_formatter_fp<wchar_t>
{};
#  if _CCCL_HAS_LONG_DOUBLE()
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<long double, wchar_t> : __fmt_formatter_fp<wchar_t>
{};
#  endif // _CCCL_HAS_LONG_DOUBLE()
#endif // _CCCL_HAS_WCHAR_T()

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD__FORMAT_FORMATERS_FP_H
