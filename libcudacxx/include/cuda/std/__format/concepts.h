//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___FORMAT_CONCEPTS_H
#define _CUDA_STD___FORMAT_CONCEPTS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__concepts/semiregular.h>
#include <cuda/std/__fwd/format.h>
#include <cuda/std/__fwd/tuple.h>
#include <cuda/std/__type_traits/remove_const.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__utility/pair.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

/// The character type specializations of \ref formatter.
template <class _CharT>
_CCCL_CONCEPT __fmt_char_type = same_as<_CharT, char>
#if _CCCL_HAS_WCHAR_T()
                             || same_as<_CharT, wchar_t>
#endif // _CCCL_HAS_WCHAR_T()
  ;

// The output iterator isn't specified. A formatter should accept any
// output_iterator. This iterator is a minimal iterator to test the concept.
// (Note testing for (w)format_context would be a valid choice, but requires
// selecting the proper one depending on the type of _CharT.)
template <class _CharT>
using __fmt_iter_for _CCCL_NODEBUG_ALIAS = _CharT*;

template <class _Tp, class _Context, class _Formatter>
_CCCL_CONCEPT __formattable_with_helper = _CCCL_REQUIRES_EXPR(
  (_Tp, _Context, _Formatter),
  _Formatter& __f,
  const _Formatter& __cf,
  _Tp&& __t,
  _Context __fc,
  basic_format_parse_context<typename _Context::char_type> __pc)(
  _Same_as(typename decltype(__pc)::iterator) __f.parse(__pc),
  _Same_as(typename _Context::iterator) __cf.format(__t, __fc));

template <class _Tp, class _Context, class _Formatter = typename _Context::template formatter_type<remove_const_t<_Tp>>>
_CCCL_CONCEPT __formattable_with = semiregular<_Formatter> && __formattable_with_helper<_Tp, _Context, _Formatter>;

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FORMAT_CONCEPTS_H
