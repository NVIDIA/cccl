//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FORMAT_FORMATERS_FP_H
#define _LIBCUDACXX___FORMAT_FORMATERS_FP_H

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

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _CharT>
struct __fmt_formatter_fp
{
  template <class _ParseCtx>
  _CCCL_API constexpr typename _ParseCtx::iterator parse(_ParseCtx& __ctx)
  {
    typename _ParseCtx::iterator __result = __parser_.__parse(__ctx, _CUDA_VSTD::__fmt_spec_fields_fp());
    _CUDA_VSTD::__fmt_process_parsed_fp(__parser_);
    return __result;
  }

  template <class _Tp, class _FmtCtx>
  _CCCL_API typename _FmtCtx::iterator format(_Tp __value, _FmtCtx& __ctx) const
  {
    _CCCL_ASSERT(false, "formatter</*floating-point-type*/>::format() is not implemented");
    (void) __value;
    return __ctx.out();
  }

private:
  __fmt_spec_parser<_CharT> __parser_;
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

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___FORMAT_FORMATERS_FP_H
