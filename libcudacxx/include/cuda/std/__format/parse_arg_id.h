//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD__FORMAT_PARSE_ARG_ID_H
#define _CUDA_STD__FORMAT_PARSE_ARG_ID_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__format/format_error.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _It>
struct __fmt_parse_number_result
{
  _It __last;
  uint32_t __value;
};

template <class _It>
_CCCL_HOST_DEVICE __fmt_parse_number_result(_It, uint32_t) -> __fmt_parse_number_result<_It>;

//! The maximum value of a numeric argument.
//!
//! This is used for:
//! - arg-id
//! - width as value or arg-id.
//! - precision as value or arg-id.
//!
//! The value is compatible with the maximum formatting width and precision
//! using the `%*` syntax on a 32-bit system.
inline constexpr uint32_t __fmt_number_max = static_cast<uint32_t>(numeric_limits<int32_t>::max());

//! Parses a number.
//! The number is used for the 31-bit values @em width and @em precision. This allows a maximum value of 2147483647.
template <class _It>
[[nodiscard]] _CCCL_API constexpr __fmt_parse_number_result<_It> __fmt_parse_number(_It __begin, _It __end_input)
{
  static_assert(__fmt_number_max == static_cast<uint32_t>(numeric_limits<int32_t>::max()),
                "The algorithm is implemented based on this value.");

  using _CharT = iter_value_t<_It>;

  // Limit the input to 9 digits, otherwise we need two checks during every
  // iteration:
  // - Are we at the end of the input?
  // - Does the value exceed width of an uint32_t? (Switching to uint64_t would
  //   have the same issue, but with a higher maximum.)
  _It __end        = __end_input - __begin > 9 ? __begin + 9 : __end_input;
  uint32_t __value = *__begin - _CharT{'0'};
  while (++__begin != __end)
  {
    if (*__begin < _CharT{'0'} || *__begin > _CharT{'9'})
    {
      return {__begin, __value};
    }
    __value = __value * 10 + *__begin - _CharT{'0'};
  }

  if (__begin != __end_input && *__begin >= _CharT{'0'} && *__begin <= _CharT{'9'})
  {
    // There are more than 9 digits, do additional validations:
    // - Does the 10th digit exceed the maximum allowed value?
    // - Are there more than 10 digits?
    // (More than 10 digits always overflows the maximum.)
    uint64_t __v = uint64_t(__value) * 10 + *__begin++ - _CharT{'0'};
    if (__v > __fmt_number_max || (__begin != __end_input && *__begin >= _CharT{'0'} && *__begin <= _CharT{'9'}))
    {
      ::cuda::std::__throw_format_error("The numeric value of the format specifier is too large");
    }
    __value = static_cast<uint32_t>(__v);
  }

  return {__begin, __value};
}

//! Multiplexer for all parse functions.
//!
//! The parser will return a pointer beyond the last consumed character. This
//! should be the closing '}' of the arg-id.
template <class _It, class _ParseCtx>
[[nodiscard]] _CCCL_API constexpr __fmt_parse_number_result<_It>
__fmt_parse_arg_id(_It __begin, _It __end, _ParseCtx& __parse_ctx)
{
  using _CharT = iter_value_t<_It>;

  switch (*__begin)
  {
    case _CharT{'0'}:
      __parse_ctx.check_arg_id(0);
      return {++__begin, 0}; // can never be larger than the maximum.

    case _CharT{':'}:
      // This case is conditionally valid. It's allowed in an arg-id in the
      // replacement-field, but not in the std-format-spec. The caller can
      // provide a better diagnostic, so accept it here unconditionally.
    case _CharT{'}'}: {
      const auto __value = __parse_ctx.next_arg_id();
      _CCCL_ASSERT(__value <= __fmt_number_max, "Compilers don't support this number of arguments");
      return {__begin, static_cast<uint32_t>(__value)};
    }
    default:
      break;
  }
  if (*__begin < _CharT{'0'} || *__begin > _CharT{'9'})
  {
    ::cuda::std::__throw_format_error("The argument index starts with an invalid character");
  }

  const auto __r = ::cuda::std::__fmt_parse_number(__begin, __end);
  __parse_ctx.check_arg_id(__r.__value);
  return __r;
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD__FORMAT_PARSE_ARG_ID_H
