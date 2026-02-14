//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD__FORMAT_FORMATERS_PTR_H
#define _CUDA_STD__FORMAT_FORMATERS_PTR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__format/format_integral.h>
#include <cuda/std/__format/format_spec_parser.h>
#include <cuda/std/__format/formatter.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

//!
//! @brief Formatter for pointer types.
//!
//! @tparam _CharT The character type used for formatting.
//!
template <class _CharT>
struct __fmt_formatter_ptr
{
  //!
  //! @brief Parses the formatting specifications for pointer types.
  //!
  //! @param __ctx The parsing context containing the format specification.
  //! @return An iterator pointing to the end of the parsed format specification.
  //!
  template <class _ParseCtx>
  _CCCL_API constexpr typename _ParseCtx::iterator parse(_ParseCtx& __ctx)
  {
    typename _ParseCtx::iterator __result = __parser_.__parse(__ctx, ::cuda::std::__fmt_spec_fields_ptr());
    ::cuda::std::__fmt_process_display_type_ptr(__parser_.__type_);
    return __result;
  }

  //!
  //! @brief Formats a pointer value according to the parsed specifications.
  //!
  //! @param __value The pointer value to format.
  //! @param __ctx The formatting context where the formatted output will be stored.
  //! @return An iterator pointing to the end of the formatted output.
  //!
  template <class _Tp, class _FmtCtx>
  _CCCL_API typename _FmtCtx::iterator format([[maybe_unused]] _Tp __value, _FmtCtx& __ctx) const
  {
    auto __specs                     = __parser_.__get_parsed_std_spec(__ctx);
    __specs.__std_.__alternate_form_ = true;
    __specs.__std_.__type_ =
      (__specs.__std_.__type_ == __fmt_spec_type::__pointer_upper_case)
        ? __fmt_spec_type::__hexadecimal_upper_case
        : __fmt_spec_type::__hexadecimal_lower_case;

    return ::cuda::std::__fmt_format_int(reinterpret_cast<uintptr_t>(__value), __ctx, __specs);
  }

private:
  __fmt_spec_parser<_CharT> __parser_; //!< The parser for format specifications.
};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<nullptr_t, char> : __fmt_formatter_ptr<char>
{};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<void*, char> : __fmt_formatter_ptr<char>
{};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<const void*, char> : __fmt_formatter_ptr<char>
{};

#if _CCCL_HAS_WCHAR_T()
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<nullptr_t, wchar_t> : __fmt_formatter_ptr<wchar_t>
{};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<void*, wchar_t> : __fmt_formatter_ptr<wchar_t>
{};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<const void*, wchar_t> : __fmt_formatter_ptr<wchar_t>
{};
#endif // _CCCL_HAS_WCHAR_T()

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD__FORMAT_FORMATERS_PTR_H
