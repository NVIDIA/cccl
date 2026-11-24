//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD__FORMAT_FORMAT_INTEGRAL_H
#define _CUDA_STD__FORMAT_FORMAT_INTEGRAL_H

#include <cuda/std/detail/__config>

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/uabs.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__charconv/to_chars.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__format/format_error.h>
#include <cuda/std/__format/format_spec_parser.h>
#include <cuda/std/__format/output_utils.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/make_nbit_int.h>
#include <cuda/std/__utility/cmp.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/to_underlying.h>
#include <cuda/std/climits>
#include <cuda/std/cstdint>
#include <cuda/std/string_view>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Tp, class _CharT, class _OutIt>
[[nodiscard]] _CCCL_HOST_DEVICE_API _OutIt
__fmt_format_char(_Tp __value, _OutIt __out_it, __fmt_parsed_spec<_CharT> __specs)
{
  if constexpr (!is_same_v<_CharT, _Tp>)
  {
    using _CharInt = __make_nbit_int_t<sizeof(_CharT) * CHAR_BIT, is_signed_v<_CharT>>;
    using _TpInt   = __make_nbit_int_t<sizeof(_Tp) * CHAR_BIT, is_signed_v<_Tp>>;

    if (!::cuda::std::in_range<_CharInt>(static_cast<_TpInt>(__value)))
    {
      ::cuda::std::__throw_format_error("Integral value outside the range of the char type");
    }
  }
  const _CharT __buffer[] = {static_cast<_CharT>(__value)};
  return ::cuda::std::__fmt_write(__buffer, __buffer + 1, ::cuda::std::move(__out_it), __specs);
}

// Share the digit conversion and its scratch buffer across signed and unsigned formatters on device.
template <class _Tp, class _CharT, class _OutIt>
[[nodiscard]] _CCCL_HOST_DEVICE_API _CCCL_NOINLINE_DEVICE _OutIt
__fmt_format_int_impl(_Tp __value, _OutIt __out_it, __fmt_parsed_spec<_CharT> __specs, bool __negative)
{
  // Binary digits need the most space. Leave room for the sign and the two-character base prefix.
  char __buffer[numeric_limits<_Tp>::digits + 3];
  char* const __last = __buffer + sizeof(__buffer);
  char* __digits     = nullptr;
  char __prefix      = '\0';
  const auto __type  = __specs.__std_.__type_;
  switch (__type)
  {
    case __fmt_spec_type::__binary_lower_case:
    case __fmt_spec_type::__binary_upper_case:
      __digits = ::cuda::std::__to_chars_int_generic(__last, __value, 2);
      __prefix = __type == __fmt_spec_type::__binary_upper_case ? 'B' : 'b';
      break;
    case __fmt_spec_type::__octal:
      __digits = ::cuda::std::__to_chars_int_generic(__last, __value, 8);
      // In alternate form octal zero is still a single digit.
      __prefix = __value != 0 ? '0' : '\0';
      break;
    case __fmt_spec_type::__default:
    case __fmt_spec_type::__decimal:
      __digits = ::cuda::std::__to_chars_int_generic(__last, __value, 10);
      break;
    case __fmt_spec_type::__hexadecimal_lower_case:
    case __fmt_spec_type::__hexadecimal_upper_case: {
      const bool __uppercase = __type == __fmt_spec_type::__hexadecimal_upper_case;
      __digits               = ::cuda::std::__to_chars_int_generic(__last, __value, 16, __uppercase);
      __prefix               = __uppercase ? 'X' : 'x';
      break;
    }
    default:
      _CCCL_UNREACHABLE();
  }

  char* __first = __digits;
  if (__specs.__std_.__alternate_form_ && __prefix != '\0')
  {
    if (__prefix != '0')
    {
      *--__first = __prefix;
    }
    *--__first = '0';
  }
  if (__negative)
  {
    *--__first = '-';
  }
  else if (__fmt_spec_sign{__specs.__std_.__sign_} == __fmt_spec_sign::__plus)
  {
    *--__first = '+';
  }
  else if (__fmt_spec_sign{__specs.__std_.__sign_} == __fmt_spec_sign::__space)
  {
    *--__first = ' ';
  }

  if (__fmt_spec_alignment{__specs.__std_.__alignment_} == __fmt_spec_alignment::__zero_padding)
  {
    // The sign and base prefix precede any zero padding.
    __out_it                    = ::cuda::std::__fmt_copy(__first, __digits, ::cuda::std::move(__out_it));
    __specs.__std_.__alignment_ = ::cuda::std::to_underlying(__fmt_spec_alignment::__right);
    __specs.__fill_.__data[0]   = _CharT{'0'};
    __specs.__width_ -= ::cuda::std::min(static_cast<uint32_t>(__digits - __first), __specs.__width_);
    __first = __digits;
  }
  return ::cuda::std::__fmt_write(__first, __last, ::cuda::std::move(__out_it), __specs);
}

template <class _Tp, class _CharT, class _OutIt>
[[nodiscard]] _CCCL_HOST_DEVICE_API _OutIt
__fmt_format_int(_Tp __value, _OutIt __out_it, __fmt_parsed_spec<_CharT> __specs)
{
  if constexpr (is_signed_v<_Tp>)
  {
    return ::cuda::std::__fmt_format_int_impl(::cuda::uabs(__value), ::cuda::std::move(__out_it), __specs, __value < 0);
  }
  else
  {
    return ::cuda::std::__fmt_format_int_impl(__value, ::cuda::std::move(__out_it), __specs, false);
  }
}

template <class _CharT, class _OutIt>
[[nodiscard]] _CCCL_HOST_DEVICE_API _OutIt
__fmt_format_bool(bool __value, _OutIt __out_it, __fmt_parsed_spec<_CharT> __specs)
{
  basic_string_view<_CharT> __str{};
  if constexpr (is_same_v<_CharT, char>)
  {
    __str = __value ? "true" : "false";
  }
#if _CCCL_HAS_WCHAR_T()
  else if constexpr (is_same_v<_CharT, wchar_t>)
  {
    __str = __value ? L"true" : L"false";
  }
#endif // _CCCL_HAS_WCHAR_T()
  else
  {
    static_assert(__always_false_v<_CharT>, "Unsupported character type for boolean formatting");
  }
  return ::cuda::std::__fmt_write(__str, ::cuda::std::move(__out_it), __specs, static_cast<ptrdiff_t>(__str.size()));
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD__FORMAT_FORMAT_INTEGRAL_H
