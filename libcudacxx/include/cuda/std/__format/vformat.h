//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FORMAT_VFORMAT_H
#define _LIBCUDACXX___FORMAT_VFORMAT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__format/format_arg.h>
#include <cuda/std/__format/format_error.h>
#include <cuda/std/__format/formatter.h>
#include <cuda/std/__format/parse_arg_id.h>
#include <cuda/std/__format/validation.h>
#include <cuda/std/__iterator/iterator_traits.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _CharT, class _ParseCtx, class _FormatCtx>
struct __fmt_replacement_field_visitor
{
  _ParseCtx& __parse_ctx_;
  _FormatCtx& __format_ctx_;
  bool __parse_;

  template <class _Tp>
  _CCCL_API constexpr void operator()(_Tp __arg)
  {
    if constexpr (is_same_v<_Tp, monostate>)
    {
      ::cuda::std::__throw_format_error("The argument index value is too large for the number of arguments supplied");
    }
    else if constexpr (is_same_v<_Tp, typename basic_format_arg<_FormatCtx>::handle>)
    {
      __arg.format(__parse_ctx_, __format_ctx_);
    }
    else
    {
      formatter<_Tp, _CharT> __formatter;
      if (__parse_)
      {
        __parse_ctx_.advance_to(__formatter.parse(__parse_ctx_));
      }
      __format_ctx_.advance_to(__formatter.format(__arg, __format_ctx_));
    }
  }
};

template <class _It, class _ParseCtx, class _Ctx>
[[nodiscard]] _CCCL_API constexpr _It
__fmt_handle_replacement_field(_It __begin, _It __end, _ParseCtx& __parse_ctx, _Ctx& __ctx)
{
  using _CharT = iter_value_t<_It>;

  const auto __r = ::cuda::std::__fmt_parse_arg_id(__begin, __end, __parse_ctx);
  if (__r.__last == __end)
  {
    ::cuda::std::__throw_format_error("The argument index should end with a ':' or a '}'");
  }

  const bool __parse = (*__r.__last == _CharT{':'});
  switch (*__r.__last)
  {
    case _CharT{':'}:
      // The arg-id has a format-specifier, advance the input to the format-spec.
      __parse_ctx.advance_to(__r.__last + 1);
      break;
    case _CharT{'}'}:
      // The arg-id has no format-specifier.
      __parse_ctx.advance_to(__r.__last);
      break;
    default:
      ::cuda::std::__throw_format_error("The argument index should end with a ':' or a '}'");
  }

  if constexpr (is_same_v<_Ctx, __fmt_validation_format_context<_CharT>>)
  {
    const __fmt_arg_t __type = __ctx.arg(__r.__value);
    if (__type == __fmt_arg_t::__none)
    {
      ::cuda::std::__throw_format_error("The argument index value is too large for the number of arguments supplied");
    }
    else if (__type == __fmt_arg_t::__handle)
    {
      __ctx.__handle(__r.__value).__parse(__parse_ctx);
    }
    else if (__parse)
    {
      ::cuda::std::__fmt_validate_visit_format_arg(__parse_ctx, __ctx, __type);
    }
  }
  else
  {
    ::cuda::std::visit_format_arg(
      __fmt_replacement_field_visitor<_CharT, _ParseCtx, _Ctx>{__parse_ctx, __ctx, __parse}, __ctx.arg(__r.__value));
  }

  __begin = __parse_ctx.begin();
  if (__begin == __end || *__begin != _CharT{'}'})
  {
    ::cuda::std::__throw_format_error("The replacement field misses a terminating '}'");
  }
  return ++__begin;
}

template <class _ParseCtx, class _Ctx>
_CCCL_API constexpr typename _Ctx::iterator __fmt_vformat_to(_ParseCtx&& __parse_ctx, _Ctx&& __ctx)
{
  using _CharT = typename _ParseCtx::char_type;
  static_assert(is_same_v<typename _Ctx::char_type, _CharT>);

  auto __begin                     = __parse_ctx.begin();
  auto __end                       = __parse_ctx.end();
  typename _Ctx::iterator __out_it = __ctx.out();
  while (__begin != __end)
  {
    switch (*__begin)
    {
      case _CharT{'{'}:
        ++__begin;
        if (__begin == __end)
        {
          ::cuda::std::__throw_format_error("The format string terminates at a '{'");
        }

        if (*__begin != _CharT{'{'})
        {
          __ctx.advance_to(::cuda::std::move(__out_it));
          __begin  = ::cuda::std::__fmt_handle_replacement_field(__begin, __end, __parse_ctx, __ctx);
          __out_it = __ctx.out();

          // The output is written and __begin points to the next character. So
          // start the next iteration.
          continue;
        }
        // The string is an escape character.
        break;

      case _CharT{'}'}:
        ++__begin;
        if (__begin == __end || *__begin != _CharT{'}'})
        {
          ::cuda::std::__throw_format_error("The format string contains an invalid escape sequence");
        }
        break;
    }

    // Copy the character to the output verbatim.
    *__out_it++ = *__begin++;
  }
  return __out_it;
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___FORMAT_VFORMAT_H
