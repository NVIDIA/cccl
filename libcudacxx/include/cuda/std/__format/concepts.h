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
#include <cuda/std/__fwd/inplace_vector.h>
#include <cuda/std/__fwd/tuple.h>
#include <cuda/std/__iterator/wrap_iter.h>
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

template <class _OutIt, class _CharT>
_CCCL_CONCEPT __fmt_enable_direct_output =
  __fmt_char_type<_CharT> && (same_as<_OutIt, _CharT*> || same_as<_OutIt, __wrap_iter<_CharT*>>);

/// Opt-in to enable \ref __fmt_insertable for a \a _Container.
template <class _Container>
inline constexpr bool __fmt_enable_insertable_v = false;

template <class _Tp, size_t _Np>
inline constexpr bool __fmt_enable_insertable_v<inplace_vector<_Tp, _Np>> = true;

// todo(dabayer): Enable insertable for host stdlib containers.

template <class _Container>
_CCCL_CONCEPT __fmt_insertable_helper = _CCCL_REQUIRES_EXPR(
  (_Container),
  _Container& __c,
  add_pointer_t<typename _Container::value_type> __first,
  add_pointer_t<typename _Container::value_type> __last)(__c.insert(__c.end(), __first, __last));

/// Concept to see whether a \a _Container is insertable.
///
/// The concept is used to validate whether multiple calls to a
/// \ref back_insert_iterator can be replace by a call to \c _Container::insert.
///
/// \note a \a _Container needs to opt-in to the concept by specializing
/// \ref __enable_insertable.
template <class _Container>
_CCCL_CONCEPT __fmt_insertable =
  __fmt_enable_insertable_v<_Container> && __fmt_char_type<typename _Container::value_type>
  && __fmt_insertable_helper<_Container>;

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FORMAT_CONCEPTS_H
