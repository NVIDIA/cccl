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

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/uabs.h>
#include <cuda/std/__charconv/to_chars.h>
#include <cuda/std/__format/format_spec_parser.h>
#include <cuda/std/__format/output_utils.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/make_nbit_int.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/__utility/cmp.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/to_underlying.h>
#include <cuda/std/climits>

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
  const auto __c = static_cast<_CharT>(__value);
  return ::cuda::std::__fmt_write(&__c, &__c + 1, ::cuda::std::move(__out_it), __specs);
}

//! Helper to determine the buffer size to output an unsigned integer in Base @em x.
//!
//! There are several overloads for the supported bases. The function uses the
//! base as template argument so it can be used in a constant expression.
template <class _Tp, int _Base>
inline constexpr int __fmt_int_buffer_size_v = 0;
template <class _Tp>
inline constexpr int __fmt_int_buffer_size_v<_Tp, 2> =
  numeric_limits<_Tp>::digits // The number of binary digits.
  + 2 // Reserve space for the '0[Bb]' prefix.
  + 1; // Reserve space for the sign.
template <class _Tp>
inline constexpr int __fmt_int_buffer_size_v<_Tp, 8> =
  numeric_limits<_Tp>::digits / 3 + 1 // The number of octal digits.
  + 1 // Reserve space for the '0' prefix.
  + 1; // Reserve space for the sign.
template <class _Tp>
inline constexpr int __fmt_int_buffer_size_v<_Tp, 10> =
  numeric_limits<_Tp>::digits10 + 1 // The number of decimal digits.
  + 1; // Reserve space for the sign.
template <class _Tp>
inline constexpr int __fmt_int_buffer_size_v<_Tp, 16> =
  numeric_limits<_Tp>::digits / 4 // The number of hexadecimal digits.
  + 2 // Reserve space for the '0x' prefix.
  + 1; // Reserve space for the sign.

[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr char*
__fmt_insert_sign(char* __buf, bool __negative, __fmt_spec_sign __sign)
{
  if (__negative)
  {
    *__buf++ = '-';
  }
  else
  {
    switch (__sign)
    {
      case __fmt_spec_sign::__default:
      case __fmt_spec_sign::__minus:
        // No sign added.
        break;
      case __fmt_spec_sign::__plus:
        *__buf++ = '+';
        break;
      case __fmt_spec_sign::__space:
        *__buf++ = ' ';
        break;
    }
  }
  return __buf;
}

template <int _Base, class _Tp, class _CharT, class _OutIt>
[[nodiscard]] _CCCL_HOST_DEVICE_API _OutIt __fmt_format_int_impl(
  _Tp __value, _OutIt __out_it, __fmt_parsed_spec<_CharT> __specs, bool __negative, const char* __prefix)
{
  constexpr auto __buffer_size = __fmt_int_buffer_size_v<_Tp, _Base>;
  char __buffer[__buffer_size];

  char* __first = ::cuda::std::__fmt_insert_sign(__buffer, __negative, __fmt_spec_sign{__specs.__std_.__sign_});
  if (__specs.__std_.__alternate_form_ && __prefix != nullptr)
  {
    while (*__prefix)
    {
      *__first++ = *__prefix++;
    }
  }

  char* __last{};
  {
    const auto __r = ::cuda::std::to_chars(__first, __buffer + __buffer_size, __value, _Base);
    _CCCL_ASSERT(__r.ec == errc(0), "Internal buffer too small");
    __last = __r.ptr;
  }

  if (__fmt_spec_alignment{__specs.__alignment_} != __fmt_spec_alignment::__zero_padding)
  {
    __first = __buffer;
  }
  else
  {
    // __buf contains [sign][prefix]data
    //                              ^ location of __first
    // The zero padding is done like:
    // - Write [sign][prefix]
    // - Write data right aligned with '0' as fill character.
    __out_it                  = ::cuda::std::__fmt_copy(__buffer, __first, ::cuda::std::move(__out_it));
    __specs.__alignment_      = ::cuda::std::to_underlying(__fmt_spec_alignment::__right);
    __specs.__fill_.__data[0] = _CharT{'0'};
    __specs.__width_ -= ::cuda::std::min(static_cast<uint32_t>(__first - __buffer), __specs.__width_);
  }

  if (__specs.__std_.__type_ != __fmt_spec_type::__hexadecimal_upper_case)
  {
    return ::cuda::std::__fmt_write(__first, __last, ::cuda::std::move(__out_it), __specs);
  }
  return ::cuda::std::__fmt_write_transformed(
    __first, __last, ::cuda::std::move(__out_it), __specs, ::cuda::std::__fmt_hex_to_upper);
}

template <class _Tp, class _CharT, class _OutIt>
[[nodiscard]] _CCCL_HOST_DEVICE_API _OutIt __fmt_format_int(
  _Tp __value, _OutIt __out_it, __fmt_parsed_spec<_CharT> __specs, [[maybe_unused]] bool __negative = false)
{
  static_assert(__cccl_is_integer_v<_Tp>);

  if constexpr (is_signed_v<_Tp>)
  {
    return ::cuda::std::__fmt_format_int(::cuda::uabs(__value), ::cuda::std::move(__out_it), __specs, __value < 0);
  }
  else
  {
    switch (__specs.__std_.__type_)
    {
      case __fmt_spec_type::__binary_lower_case:
        return ::cuda::std::__fmt_format_int_impl<2>(__value, ::cuda::std::move(__out_it), __specs, __negative, "0b");
      case __fmt_spec_type::__binary_upper_case:
        return ::cuda::std::__fmt_format_int_impl<2>(__value, ::cuda::std::move(__out_it), __specs, __negative, "0B");
      case __fmt_spec_type::__octal:
        // Octal is special; if __value == 0 there's no prefix.
        return ::cuda::std::__fmt_format_int_impl<8>(
          __value, ::cuda::std::move(__out_it), __specs, __negative, __value != 0 ? "0" : nullptr);
      case __fmt_spec_type::__default:
      case __fmt_spec_type::__decimal:
        return ::cuda::std::__fmt_format_int_impl<10>(
          __value, ::cuda::std::move(__out_it), __specs, __negative, nullptr);
      case __fmt_spec_type::__hexadecimal_lower_case:
        return ::cuda::std::__fmt_format_int_impl<16>(__value, ::cuda::std::move(__out_it), __specs, __negative, "0x");
      case __fmt_spec_type::__hexadecimal_upper_case:
        return ::cuda::std::__fmt_format_int_impl<16>(__value, ::cuda::std::move(__out_it), __specs, __negative, "0X");
      default:
        _CCCL_UNREACHABLE();
    }
  }
}

template <class _CharT, class _OutIt>
[[nodiscard]] _CCCL_HOST_DEVICE_API _OutIt
__fmt_format_bool(bool __value, _OutIt __out_it, __fmt_parsed_spec<_CharT> __specs)
{
  basic_string_view<_CharT> __str{};
  if constexpr (is_same_v<_CharT, char>)
  {
    __str = (__value) ? "true" : "false";
  }
#if _CCCL_HAS_WCHAR_T()
  else if constexpr (is_same_v<_CharT, wchar_t>)
  {
    __str = (__value) ? L"true" : L"false";
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
