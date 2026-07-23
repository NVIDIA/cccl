//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___FORMAT_FORMATERS_TUPLE_H
#define _CUDA_STD___FORMAT_FORMATERS_TUPLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__utility/static_for.h>
#include <cuda/std/__algorithm/copy.h>
#include <cuda/std/__format/buffer.h>
#include <cuda/std/__format/concepts.h>
#include <cuda/std/__format/format_context.h>
#include <cuda/std/__format/format_error.h>
#include <cuda/std/__format/format_spec_parser.h>
#include <cuda/std/__format/formatter.h>
#include <cuda/std/__format/output_utils.h>
#include <cuda/std/__fwd/pair.h>
#include <cuda/std/__fwd/tuple.h>
#include <cuda/std/__string/literal.h>
#include <cuda/std/__tuple_dir/get.h>
#include <cuda/std/__tuple_dir/tuple.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/fold.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/to_underlying.h>
#include <cuda/std/string_view>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _CharT, class _Tuple, class _Void, class... _Args>
class __fmt_formatter_tuple : public __fmt_disabled_formatter
{};

template <class _CharT>
[[nodiscard]] _CCCL_API constexpr __fmt_spec_parser<_CharT> __fmt_formatter_tuple_make_parser() noexcept
{
  __fmt_spec_parser<_CharT> __parser{};
  __parser.__alignment_ = ::cuda::std::to_underlying(__fmt_spec_alignment::__left);
  return __parser;
}

template <class _CharT, class _Tuple, class... _Args>
class __fmt_formatter_tuple<_CharT,
                            _Tuple,
                            enable_if_t<__fmt_char_type<_CharT> && __fold_and_v<formattable<_Args, _CharT>...>>,
                            _Args...>
{
  tuple<formatter<remove_cvref_t<_Args>, _CharT>...> __underlying_;
  basic_string_view<_CharT> __separator_       = _CCCL_STRLIT(_CharT, ", ");
  basic_string_view<_CharT> __opening_bracket_ = _CCCL_STRLIT(_CharT, "(");
  basic_string_view<_CharT> __closing_bracket_ = _CCCL_STRLIT(_CharT, ")");

  template <class _Tuple2, class _FormatContext>
  [[nodiscard]] _CCCL_API typename _FormatContext::iterator
  __format_tuple(_Tuple2&& __tuple, _FormatContext& __ctx) const
  {
    __ctx.advance_to(::cuda::std::copy(__opening_bracket_.begin(), __opening_bracket_.end(), __ctx.out()));

    ::cuda::static_for<sizeof...(_Args)>([&](auto __i) {
      using ::cuda::std::get;

      if constexpr (__i() > 0)
      {
        __ctx.advance_to(::cuda::std::copy(__separator_.begin(), __separator_.end(), __ctx.out()));
      }
      __ctx.advance_to(get<__i()>(__underlying_).format(get<__i()>(__tuple), __ctx));
    });

    return ::cuda::std::copy(__closing_bracket_.begin(), __closing_bracket_.end(), __ctx.out());
  }

public:
  __fmt_spec_parser<_CharT> __parser_ = ::cuda::std::__fmt_formatter_tuple_make_parser<_CharT>();

  _CCCL_API constexpr void set_separator(basic_string_view<_CharT> __separator) noexcept
  {
    __separator_ = __separator;
  }
  _CCCL_API constexpr void
  set_brackets(basic_string_view<_CharT> __opening_bracket, basic_string_view<_CharT> __closing_bracket) noexcept
  {
    __opening_bracket_ = __opening_bracket;
    __closing_bracket_ = __closing_bracket;
  }

  template <class _ParseContext>
  _CCCL_API constexpr typename _ParseContext::iterator parse(_ParseContext& __ctx)
  {
    auto __begin = __parser_.__parse(__ctx, ::cuda::std::__fmt_spec_fields_tuple());
    auto __end   = __ctx.end();

    // Note 'n' is part of the type here
    if (__parser_.__clear_brackets_)
    {
      set_brackets({}, {});
    }
    else if (__begin != __end && *__begin == _CharT{'m'})
    {
      if constexpr (sizeof...(_Args) == 2)
      {
        set_separator(_CCCL_STRLIT(_CharT, ": "));
        set_brackets({}, {});
        ++__begin;
      }
      else
      {
        ::cuda::std::__throw_format_error("Type m requires a pair or a tuple with two elements");
      }
    }

    if (__begin != __end && *__begin != _CharT{'}'})
    {
      ::cuda::std::__throw_format_error("The format specifier should consume the input or end with a '}'");
    }

    __ctx.advance_to(__begin);

    // [format.tuple]/7
    //   ... For each element e in underlying_, if e.set_debug_format()
    //   is a valid expression, calls e.set_debug_format().
    ::cuda::static_for<sizeof...(_Args)>([&](auto __i) {
      using ::cuda::std::get;
      auto& __formatter = get<__i()>(__underlying_);
      __formatter.parse(__ctx);

      // todo(dabayer): Enable setting debug format once we have debug format support.
      // Unlike the range_formatter we don't guard against evil parsers. Since
      // this format-spec never has a format-spec for the underlying type
      // adding the test would give additional overhead.
      // ::cuda::std::__set_debug_format(__formatter);
    });
    return __begin;
  }

  template <class _FormatContext>
  typename _FormatContext::iterator _CCCL_API
  format(conditional_t<(formattable<const _Args, _CharT> && ...), const _Tuple&, _Tuple&> __tuple,
         _FormatContext& __ctx) const
  {
    __fmt_parsed_spec<_CharT> __specs = __parser_.__get_parsed_std_spec(__ctx);

    if (!__specs.__has_width())
    {
      return __format_tuple(__tuple, __ctx);
    }

    // The size of the buffer needed is:
    // - open bracket characters
    // - close bracket character
    // - n elements where every element may have a different size
    // - (n -1) separators
    // The size of the element is hard to predict, knowing the type helps but
    // it depends on the format-spec. As an initial estimate we guess 6
    // characters.
    // Typically both brackets are 1 character and the separator is 2
    // characters. Which means there will be
    //   (n - 1) * 2 + 1 + 1 = n * 2 character
    // So estimate 8 times the range size as buffer.
    __fmt_retarget_buffer<_CharT> __buffer{8 * tuple_size_v<_Tuple>};
    basic_format_context<typename __fmt_retarget_buffer<_CharT>::__iterator, _CharT> __c{
      __buffer.__make_output_iterator(), __ctx};

    (void) __format_tuple(__tuple, __c);
    return ::cuda::std::__fmt_write_string_no_precision(__buffer.__view(), __ctx.out(), __specs);
  }
};

template <class _CharT, class... _Args>
struct formatter<pair<_Args...>, _CharT> : __fmt_formatter_tuple<_CharT, pair<_Args...>, void, _Args...>
{};

template <class _CharT, class... _Args>
struct formatter<tuple<_Args...>, _CharT> : __fmt_formatter_tuple<_CharT, tuple<_Args...>, void, _Args...>
{};

#if _CCCL_HAS_HOST_STD_LIB()
template <class _CharT, class... _Args>
struct formatter<::std::pair<_Args...>, _CharT> : __fmt_formatter_tuple<_CharT, ::std::pair<_Args...>, void, _Args...>
{};

template <class _CharT, class... _Args>
struct formatter<::std::tuple<_Args...>, _CharT> : __fmt_formatter_tuple<_CharT, ::std::tuple<_Args...>, void, _Args...>
{};
#endif // _CCCL_HAS_HOST_STD_LIB()

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FORMAT_FORMATERS_TUPLE_H
