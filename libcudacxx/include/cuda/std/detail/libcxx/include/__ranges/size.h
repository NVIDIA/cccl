// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCUDACXX___RANGES_SIZE_H
#define _LIBCUDACXX___RANGES_SIZE_H

#ifndef __cuda_std__
#  include <__config>
#endif // __cuda_std__

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include "../__concepts/arithmetic.h"
#include "../__concepts/class_or_enum.h"
#include "../__iterator/concepts.h"
#include "../__iterator/iterator_traits.h"
#include "../__memory/pointer_traits.h"
#include "../__ranges/access.h"
#include "../__type_traits/is_unbounded_array.h"
#include "../__type_traits/make_unsigned.h"
#include "../__type_traits/make_signed.h"
#include "../__type_traits/remove_cvref.h"
#include "../__utility/auto_cast.h"
#include "../__utility/declval.h"
#include "../cstddef"
#include "../cstdlib"

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES

#if _CCCL_STD_VER >= 2017 && !defined(_CCCL_COMPILER_MSVC_2017)

template <class>
_LIBCUDACXX_INLINE_VAR constexpr bool disable_sized_range = false;

// [range.prim.size]

_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__size)
template <class _Tp>
void size(_Tp&) = delete;
template <class _Tp>
void size(const _Tp&) = delete;

template <class _Tp>
_LIBCUDACXX_CONCEPT __size_enabled = !disable_sized_range<remove_cvref_t<_Tp>>;

#  if _CCCL_STD_VER >= 2020
template <class _Tp>
concept __member_size = __size_enabled<_Tp> && __workaround_52970<_Tp> && requires(_Tp&& __t) {
  {
    _LIBCUDACXX_AUTO_CAST(__t.size())
  } -> __integer_like;
};

template <class _Tp>
concept __unqualified_size =
  __size_enabled<_Tp> && !__member_size<_Tp> && __class_or_enum<remove_cvref_t<_Tp>> && requires(_Tp&& __t) {
    {
      _LIBCUDACXX_AUTO_CAST(size(__t))
    } -> __integer_like;
  };

template <class _Tp>
concept __difference =
  !__member_size<_Tp> && !__unqualified_size<_Tp> && __class_or_enum<remove_cvref_t<_Tp>> && requires(_Tp&& __t) {
    {
      _CUDA_VRANGES::begin(__t)
    } -> forward_iterator;
    {
      _CUDA_VRANGES::end(__t)
    } -> sized_sentinel_for<decltype(_CUDA_VRANGES::begin(_CUDA_VSTD::declval<_Tp>()))>;
  };
#  else // ^^^ CXX20 ^^^ / vvv CXX17 vvv
template <class _Tp>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __member_size_,
  requires(_Tp&& __t)(requires(__size_enabled<_Tp>),
                      requires(__workaround_52970<_Tp>),
                      requires(__integer_like<decltype(_LIBCUDACXX_AUTO_CAST(__t.size()))>)));

template <class _Tp>
_LIBCUDACXX_CONCEPT __member_size = _LIBCUDACXX_FRAGMENT(__member_size_, _Tp);

template <class _Tp>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __unqualified_size_,
  requires(_Tp&& __t)(requires(__size_enabled<_Tp>),
                      requires(!__member_size<_Tp>),
                      requires(__class_or_enum<remove_cvref_t<_Tp>>),
                      requires(__integer_like<decltype(_LIBCUDACXX_AUTO_CAST(size(__t)))>)));

template <class _Tp>
_LIBCUDACXX_CONCEPT __unqualified_size = _LIBCUDACXX_FRAGMENT(__unqualified_size_, _Tp);

template <class _Tp>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __difference_,
  requires(_Tp&& __t)(requires(!__member_size<_Tp>),
                      requires(!__unqualified_size<_Tp>),
                      requires(__class_or_enum<remove_cvref_t<_Tp>>),
                      requires(forward_iterator<decltype(_CUDA_VRANGES::begin(__t))>),
                      requires(sized_sentinel_for<decltype(_CUDA_VRANGES::end(__t)),
                                                  decltype(_CUDA_VRANGES::begin(_CUDA_VSTD::declval<_Tp>()))>)));

template <class _Tp>
_LIBCUDACXX_CONCEPT __difference = _LIBCUDACXX_FRAGMENT(__difference_, _Tp);
#  endif // _CCCL_STD_VER <= 2017

struct __fn
{
  // `[range.prim.size]`: the array case (for rvalues).
  template <class _Tp, size_t _Sz>
  _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr size_t
  operator()(_Tp (&&)[_Sz]) const noexcept
  {
    return _Sz;
  }

  // `[range.prim.size]`: the array case (for lvalues).
  template <class _Tp, size_t _Sz>
  _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr size_t
  operator()(_Tp (&)[_Sz]) const noexcept
  {
    return _Sz;
  }

  // `[range.prim.size]`: `auto(t.size())` is a valid expression.
  _LIBCUDACXX_TEMPLATE(class _Tp)
  _LIBCUDACXX_REQUIRES(__member_size<_Tp>)
  _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr auto
  operator()(_Tp&& __t) const noexcept(noexcept(_LIBCUDACXX_AUTO_CAST(__t.size())))
  {
    return _LIBCUDACXX_AUTO_CAST(__t.size());
  }

  // `[range.prim.size]`: `auto(size(t))` is a valid expression.
  _LIBCUDACXX_TEMPLATE(class _Tp)
  _LIBCUDACXX_REQUIRES(__unqualified_size<_Tp>)
  _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr auto
  operator()(_Tp&& __t) const noexcept(noexcept(_LIBCUDACXX_AUTO_CAST(size(__t))))
  {
    return _LIBCUDACXX_AUTO_CAST(size(__t));
  }

  // [range.prim.size]: the `to-unsigned-like` case.
  _LIBCUDACXX_TEMPLATE(class _Tp)
  _LIBCUDACXX_REQUIRES(__difference<_Tp>)
  _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr auto
  operator()(_Tp&& __t) const
    noexcept(noexcept(_CUDA_VSTD::__to_unsigned_like(_CUDA_VRANGES::end(__t) - _CUDA_VRANGES::begin(__t))))
      -> decltype(_CUDA_VSTD::__to_unsigned_like(_CUDA_VRANGES::end(__t) - _CUDA_VRANGES::begin(__t)))
  {
    return _CUDA_VSTD::__to_unsigned_like(_CUDA_VRANGES::end(__t) - _CUDA_VRANGES::begin(__t));
  }
};
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo
{
_LIBCUDACXX_CPO_ACCESSIBILITY auto size = __size::__fn{};
} // namespace __cpo

// [range.prim.ssize]

_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__ssize)
#  if _CCCL_STD_VER >= 2020
template <class _Tp>
concept __can_ssize = requires(_Tp&& __t) { _CUDA_VRANGES::size(__t); };
#  else // ^^^ CXX20 ^^^ / vvv CXX17 vvv
template <class _Tp>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __can_ssize_, requires(_Tp&& __t)(requires(!is_unbounded_array_v<_Tp>), ((void) _CUDA_VRANGES::size(__t))));

template <class _Tp>
_LIBCUDACXX_CONCEPT __can_ssize = _LIBCUDACXX_FRAGMENT(__can_ssize_, _Tp);
#  endif // _CCCL_STD_VER <= 2017

struct __fn
{
  _LIBCUDACXX_TEMPLATE(class _Tp)
  _LIBCUDACXX_REQUIRES(__can_ssize<_Tp>)
  _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr auto
  operator()(_Tp&& __t) const noexcept(noexcept(_CUDA_VRANGES::size(__t)))
  {
    using _Signed = make_signed_t<decltype(_CUDA_VRANGES::size(__t))>;
    if constexpr (sizeof(ptrdiff_t) > sizeof(_Signed))
    {
      return static_cast<ptrdiff_t>(_CUDA_VRANGES::size(__t));
    }
    else
    {
      return static_cast<_Signed>(_CUDA_VRANGES::size(__t));
    }
    _LIBCUDACXX_UNREACHABLE();
  }
};
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo
{
_LIBCUDACXX_CPO_ACCESSIBILITY auto ssize = __ssize::__fn{};
} // namespace __cpo

#endif // _CCCL_STD_VER >= 2017 && !_CCCL_COMPILER_MSVC_2017

_LIBCUDACXX_END_NAMESPACE_RANGES

#endif // _LIBCUDACXX___RANGES_SIZE_H
