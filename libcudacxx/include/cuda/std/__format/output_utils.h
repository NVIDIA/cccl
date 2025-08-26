//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD__FORMAT_OUTPUT_UTILS_H
#define _CUDA_STD__FORMAT_OUTPUT_UTILS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/copy.h>
#include <cuda/std/__algorithm/fill_n.h>
#include <cuda/std/__algorithm/transform.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__format/format_spec_parser.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/string_view>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

[[nodiscard]] _CCCL_API constexpr char __fmt_hex_to_upper(char __c) noexcept
{
  switch (__c)
  {
    case 'a':
      return 'A';
    case 'b':
      return 'B';
    case 'c':
      return 'C';
    case 'd':
      return 'D';
    case 'e':
      return 'E';
    case 'f':
      return 'F';
    default:
      return __c;
  }
}

struct __fmt_padding_size_result
{
  size_t __before_;
  size_t __after_;
};

// nvcc warns about missing return statement when compiling with msvc host compiler, adding more unreachables doesn't
// help, so let's just suppress the warning
#if _CCCL_COMPILER(MSVC)
_CCCL_BEGIN_NV_DIAG_SUPPRESS(940) // missing return statement at end of non-void function
#endif // _CCCL_COMPILER(MSVC)

[[nodiscard]] _CCCL_API constexpr __fmt_padding_size_result
__fmt_padding_size(size_t __size, size_t __width, __fmt_spec_alignment __align)
{
  _CCCL_ASSERT(__width > __size, "don't call this function when no padding is required");
  _CCCL_ASSERT(__align != __fmt_spec_alignment::__zero_padding, "the caller should have handled the zero-padding");

  const size_t __fill = __width - __size;
  switch (__align)
  {
    case __fmt_spec_alignment::__left:
      return {0, __fill};
    case __fmt_spec_alignment::__center: {
      // The extra padding is divided per [format.string.std]/3
      // __before = floor(__fill, 2);
      // __after = ceil(__fill, 2);
      const size_t __before = __fill / 2;
      const size_t __after  = __fill - __before;
      return {__before, __after};
    }
    case __fmt_spec_alignment::__default:
    case __fmt_spec_alignment::__right:
      return {__fill, 0};
    case __fmt_spec_alignment::__zero_padding:
    default:
      _CCCL_UNREACHABLE();
  }
}

#if _CCCL_COMPILER(MSVC)
_CCCL_END_NV_DIAG_SUPPRESS()
#endif // _CCCL_COMPILER(MSVC)

//! Copy wrapper.
//!
//! This uses a "mass output function" of __format::__output_buffer when possible.
template <class _CharT, class _OutCharT = _CharT, class _OutIt>
[[nodiscard]] _CCCL_API _OutIt __fmt_copy(basic_string_view<_CharT> __str, _OutIt __out_it)
{
  // todo: handle __fmt_output_buffer and __fmt_retarget_buffer when they are implemented
  return ::cuda::std::copy(__str.begin(), __str.end(), ::cuda::std::move(__out_it));
}

template <class _It, class _CharT = iter_value_t<_It>, class _OutCharT = _CharT, class _OutIt>
[[nodiscard]] _CCCL_API _OutIt __fmt_copy(_It __first, _It __last, _OutIt __out_it)
{
  return ::cuda::std::__fmt_copy(basic_string_view{__first, __last}, ::cuda::std::move(__out_it));
}

template <class _It, class _CharT = iter_value_t<_It>, class _OutCharT = _CharT, class _OutIt>
[[nodiscard]] _CCCL_API _OutIt __fmt_copy(_It __first, size_t __n, _OutIt __out_it)
{
  return ::cuda::std::__fmt_copy(basic_string_view{::cuda::std::to_address(__first), __n}, ::cuda::std::move(__out_it));
}

//! Transform wrapper.
//!
//! This uses a "mass output function" of __format::__output_buffer when possible.
template <class _It, class _CharT = iter_value_t<_It>, class _OutCharT = _CharT, class _OutIt, class _UnaryOp>
[[nodiscard]] _CCCL_API _OutIt __fmt_transform(_It __first, _It __last, _OutIt __out_it, _UnaryOp __operation)
{
  // todo: handle __fmt_output_buffer and __fmt_retarget_buffer when they are implemented
  return ::cuda::std::transform(__first, __last, ::cuda::std::move(__out_it), __operation);
}

//! Fill wrapper.
//!
//! This uses a "mass output function" of __format::__output_buffer when possible.
template <class _CharT, class _OutIt>
[[nodiscard]] _CCCL_API _OutIt __fmt_fill(_OutIt __out_it, size_t __n, _CharT __value)
{
  // todo: handle __fmt_output_buffer and __fmt_retarget_buffer when they are implemented
  return ::cuda::std::fill_n(::cuda::std::move(__out_it), __n, __value);
}

template <class _CharT, class _OutIt>
[[nodiscard]] _CCCL_API _OutIt __fmt_fill(_OutIt __out_it, size_t __n, __fmt_spec_code_point<_CharT> __value)
{
  return ::cuda::std::__fmt_fill(::cuda::std::move(__out_it), __n, __value.__data[0]);
}

//! Writes the input to the output with the required padding.
//!
//! Since the output column width is specified the function can be used for
//! ASCII and Unicode output.
//!
//! @pre \a __size <= \a __width. Using this function when this pre-condition
//!      doesn't hold incurs an unwanted overhead.
//!
//! @param __str       The string to write.
//! @param __out_it    The output iterator to write to.
//! @param __specs     The parsed formatting specifications.
//! @param __size      The (estimated) output column width. When the elements
//!                    to be written are ASCII the following condition holds
//!                    \a __size == \a __last - \a __first.
//!
//! @returns           An iterator pointing beyond the last element written.
//!
//! @note The type of the elements in range [\a __first, \a __last) can differ
//! from the type of \a __specs. Integer output uses \c std::to_chars for its
//! conversion, which means the [\a __first, \a __last) always contains elements
//! of the type \c char.
template <class _CharT, class _ParserCharT, class _OutIt>
[[nodiscard]] _CCCL_API _OutIt
__fmt_write(basic_string_view<_CharT> __str, _OutIt __out_it, __fmt_parsed_spec<_ParserCharT> __specs, ptrdiff_t __size)
{
  if (__size >= static_cast<ptrdiff_t>(__specs.__width_))
  {
    return ::cuda::std::__fmt_copy(__str, ::cuda::std::move(__out_it));
  }

  const auto __padding =
    ::cuda::std::__fmt_padding_size(__size, __specs.__width_, __fmt_spec_alignment{__specs.__std_.__alignment_});
  __out_it = ::cuda::std::__fmt_fill(::cuda::std::move(__out_it), __padding.__before_, __specs.__fill_);
  __out_it = ::cuda::std::__fmt_copy(__str, ::cuda::std::move(__out_it));
  return ::cuda::std::__fmt_fill(::cuda::std::move(__out_it), __padding.__after_, __specs.__fill_);
}

template <class _It, class _ParserCharT, class _OutIt>
[[nodiscard]] _CCCL_API _OutIt
__fmt_write(_It __first, _It __last, _OutIt __out_it, __fmt_parsed_spec<_ParserCharT> __specs, ptrdiff_t __size)
{
  _CCCL_ASSERT(__first <= __last, "Not a valid range");
  return ::cuda::std::__fmt_write(basic_string_view{__first, __last}, ::cuda::std::move(__out_it), __specs, __size);
}

// Calls the function above where \a __size = \a __last - \a __first.
template <class _It, class _ParserCharT, class _OutIt>
[[nodiscard]] _CCCL_API _OutIt
__fmt_write(_It __first, _It __last, _OutIt __out_it, __fmt_parsed_spec<_ParserCharT> __specs)
{
  _CCCL_ASSERT(__first <= __last, "Not a valid range");
  return ::cuda::std::__fmt_write(__first, __last, ::cuda::std::move(__out_it), __specs, __last - __first);
}

template <class _It, class _CharT = iter_value_t<_It>, class _ParserCharT, class _OutIt, class _UnaryOp>
[[nodiscard]] _CCCL_API _OutIt __fmt_write_transformed(
  _It __first, _It __last, _OutIt __out_it, __fmt_parsed_spec<_ParserCharT> __specs, _UnaryOp __op)
{
  _CCCL_ASSERT(__first <= __last, "Not a valid range");

  ptrdiff_t __size = __last - __first;
  if (__size >= __specs.__width_)
  {
    return ::cuda::std::__fmt_transform(__first, __last, ::cuda::std::move(__out_it), __op);
  }
  const auto __padding =
    ::cuda::std::__fmt_padding_size(__size, __specs.__width_, __fmt_spec_alignment{__specs.__alignment_});
  __out_it = ::cuda::std::__fmt_fill(::cuda::std::move(__out_it), __padding.__before_, __specs.__fill_);
  __out_it = ::cuda::std::__fmt_transform(__first, __last, ::cuda::std::move(__out_it), __op);
  return ::cuda::std::__fmt_fill(::cuda::std::move(__out_it), __padding.__after_, __specs.__fill_);
}

//! Writes a string using format's width estimation algorithm.
//!
//! @pre !__specs.__has_precision()
//!
//! @note When \c _LIBCPP_HAS_UNICODE is false the function assumes the input is ASCII.
template <class _CharT, class _OutIt>
[[nodiscard]] _CCCL_API _OutIt
__fmt_write_string_no_precision(basic_string_view<_CharT> __str, _OutIt __out_it, __fmt_parsed_spec<_CharT> __specs)
{
  _CCCL_ASSERT(!__specs.__has_precision(), "use __write_string");

  // No padding -> copy the string
  if (!__specs.__has_width())
  {
    return ::cuda::std::__fmt_copy(__str, ::cuda::std::move(__out_it));
  }

  // Note when the estimated width is larger than size there's no padding. So
  // there's no reason to get the real size when the estimate is larger than or
  // equal to the minimum field width.
  size_t __size =
    ::cuda::std::__fmt_estimate_column_width(__str, __specs.__width_, __fmt_column_width_rounding::__up).__width_;
  return ::cuda::std::__fmt_write(__str, ::cuda::std::move(__out_it), __specs, __size);
}

template <class _CharT>
[[nodiscard]] _CCCL_API int __fmt_truncate(basic_string_view<_CharT>& __str, int __precision)
{
  const auto __result =
    ::cuda::std::__fmt_estimate_column_width(__str, __precision, __fmt_column_width_rounding::__down);
  __str = basic_string_view<_CharT>{__str.begin(), __result.__last_};
  return static_cast<int>(__result.__width_);
}

//! Writes a string using format's width estimation algorithm.
template <class _CharT, class _OutIt>
[[nodiscard]] _CCCL_API _OutIt
__fmt_write_string(basic_string_view<_CharT> __str, _OutIt __out_it, __fmt_parsed_spec<_CharT> __specs)
{
  if (!__specs.__has_precision())
  {
    return ::cuda::std::__fmt_write_string_no_precision(__str, ::cuda::std::move(__out_it), __specs);
  }
  int __size = ::cuda::std::__fmt_truncate(__str, __specs.__precision_);
  return ::cuda::std::__fmt_write(__str.begin(), __str.end(), ::cuda::std::move(__out_it), __specs, __size);
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD__FORMAT_OUTPUT_UTILS_H
