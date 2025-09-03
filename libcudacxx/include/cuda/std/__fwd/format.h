//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___FWD_FORMAT_H
#define _CUDA_STD___FWD_FORMAT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__fwd/iterator.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _CharT>
class _CCCL_TYPE_VISIBILITY_DEFAULT basic_format_parse_context;

using format_parse_context = basic_format_parse_context<char>;
#if _CCCL_HAS_WCHAR_T()
using wformat_parse_context = basic_format_parse_context<wchar_t>;
#endif // _CCCL_HAS_WCHAR_T()

template <class _CharT>
class _CCCL_TYPE_VISIBILITY_DEFAULT _CCCL_PREFERRED_NAME(format_parse_context)
#if _CCCL_HAS_WCHAR_T()
  _CCCL_PREFERRED_NAME(wformat_parse_context)
#endif // _CCCL_HAS_WCHAR_T()
    basic_format_parse_context;

template <class _CharT>
class __fmt_output_buffer;

template <class _Context>
class _CCCL_TYPE_VISIBILITY_DEFAULT basic_format_arg;

template <class _OutIt, class _CharT>
class _CCCL_TYPE_VISIBILITY_DEFAULT basic_format_context;

using format_context = basic_format_context<__back_insert_iterator<__fmt_output_buffer<char>>, char>;
#if _CCCL_HAS_WCHAR_T()
using wformat_context = basic_format_context<__back_insert_iterator<__fmt_output_buffer<wchar_t>>, wchar_t>;
#endif // _CCCL_HAS_WCHAR_T()

template <class _OutIt, class _CharT>
class _CCCL_TYPE_VISIBILITY_DEFAULT _CCCL_PREFERRED_NAME(format_context)
#if _CCCL_HAS_WCHAR_T()
  _CCCL_PREFERRED_NAME(wformat_context)
#endif // _CCCL_HAS_WCHAR_T()
    basic_format_context;

template <class _Context>
class _CCCL_TYPE_VISIBILITY_DEFAULT basic_format_args;

using format_args = basic_format_args<format_context>;
#if _CCCL_HAS_WCHAR_T()
using wformat_args = basic_format_args<wformat_context>;
#endif // _CCCL_HAS_WCHAR_T()

template <class _Context>
class _CCCL_TYPE_VISIBILITY_DEFAULT _CCCL_PREFERRED_NAME(format_args)
#if _CCCL_HAS_WCHAR_T()
  _CCCL_PREFERRED_NAME(wformat_args)
#endif // _CCCL_HAS_WCHAR_T()
    basic_format_args;

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FWD_FORMAT_H
