// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCUDACXX___RANGES_ACCESS_H
#define _LIBCUDACXX___RANGES_ACCESS_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include "../__concepts/class_or_enum.h"
#include "../__iterator/concepts.h"
#include "../__iterator/readable_traits.h"
#include "../__ranges/enable_borrowed_range.h"
#include "../__type_traits/is_array.h"
#include "../__type_traits/is_reference.h"
#include "../__type_traits/remove_cvref.h"
#include "../__type_traits/remove_reference.h"
#include "../__utility/auto_cast.h"
#include "../__utility/declval.h"

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES

#if _CCCL_STD_VER > 2014 && !defined(_CCCL_COMPILER_MSVC_2017)

  template <class _Tp>
  _LIBCUDACXX_CONCEPT __can_borrow =
    is_lvalue_reference_v<_Tp> || enable_borrowed_range<remove_cvref_t<_Tp>>;

// [range.access.begin]

_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__begin)
  template<class _Tp>
  void begin(_Tp&) = delete;
  template<class _Tp>
  void begin(const _Tp&) = delete;

#if _CCCL_STD_VER > 2017
  template <class _Tp>
  concept __member_begin =
    __can_borrow<_Tp> &&
    __workaround_52970<_Tp> &&
    requires(_Tp&& __t) {
      { _LIBCUDACXX_AUTO_CAST(__t.begin()) } -> input_or_output_iterator;
    };

  template <class _Tp>
  concept __unqualified_begin =
    !__member_begin<_Tp> &&
    __can_borrow<_Tp> &&
    __class_or_enum<remove_cvref_t<_Tp>> &&
    requires(_Tp && __t) {
      { _LIBCUDACXX_AUTO_CAST(begin(__t)) } -> input_or_output_iterator;
    };
#else // ^^^ CXX20 ^^^ / vvv CXX17 vvv
  template<class _Tp>
  _LIBCUDACXX_CONCEPT_FRAGMENT(
    __member_begin_,
    requires(_Tp&& __t)(
      requires(__can_borrow<_Tp>),
      requires(__workaround_52970<_Tp>),
      requires(input_or_output_iterator<decltype(_LIBCUDACXX_AUTO_CAST(__t.begin()))>)
    ));

  template<class _Tp>
  _LIBCUDACXX_CONCEPT __member_begin = _LIBCUDACXX_FRAGMENT(__member_begin_, _Tp);

  template<class _Tp>
  _LIBCUDACXX_CONCEPT_FRAGMENT(
    __unqualified_begin_,
    requires(_Tp&& __t)(
      requires(!__member_begin<_Tp>),
      requires(__can_borrow<_Tp>),
      requires(__class_or_enum<remove_cvref_t<_Tp>>),
      requires(input_or_output_iterator<decltype(_LIBCUDACXX_AUTO_CAST(begin(__t)))>)
    ));

  template<class _Tp>
  _LIBCUDACXX_CONCEPT __unqualified_begin = _LIBCUDACXX_FRAGMENT(__unqualified_begin_, _Tp);
#endif // _CCCL_STD_VER < 2020

  struct __fn {
    // This has been made valid as a defect report for C++17 onwards, however gcc below 11.0 does not implement it
#if (!defined(_CCCL_COMPILER_GCC) || __GNUC__ >= 11)
    _LIBCUDACXX_TEMPLATE(class _Tp)
      _LIBCUDACXX_REQUIRES((sizeof(_Tp) >= 0)) // Disallow incomplete element types.
    _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr auto operator()(_Tp (&__t)[]) const noexcept
    {
      return __t + 0;
    }
#endif // (!defined(__GNUC__) || __GNUC__ >= 11)

    _LIBCUDACXX_TEMPLATE(class _Tp, size_t _Np)
      _LIBCUDACXX_REQUIRES((sizeof(_Tp) >= 0)) // Disallow incomplete element types.
    _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr auto operator()(_Tp (&__t)[_Np]) const noexcept
    {
      return __t + 0;
    }

    _LIBCUDACXX_TEMPLATE(class _Tp)
      _LIBCUDACXX_REQUIRES(__member_begin<_Tp>)
    _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr auto operator()(_Tp&& __t) const
      noexcept(noexcept(_LIBCUDACXX_AUTO_CAST(__t.begin())))
    {
      return _LIBCUDACXX_AUTO_CAST(__t.begin());
    }

    _LIBCUDACXX_TEMPLATE(class _Tp)
      _LIBCUDACXX_REQUIRES(__unqualified_begin<_Tp>)
    _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr auto operator()(_Tp&& __t) const
      noexcept(noexcept(_LIBCUDACXX_AUTO_CAST(begin(__t))))
    {
      return _LIBCUDACXX_AUTO_CAST(begin(__t));
    }

    _LIBCUDACXX_TEMPLATE(class _Tp)
      _LIBCUDACXX_REQUIRES((!__member_begin<_Tp>) _LIBCUDACXX_AND
                (!__unqualified_begin<_Tp>))
    void operator()(_Tp&&) const = delete;
  };
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo {
  _LIBCUDACXX_CPO_ACCESSIBILITY auto begin = __begin::__fn{};
} // namespace __cpo

// [range.range]

  template <class _Tp>
  using iterator_t = decltype(_CUDA_VRANGES::begin(_CUDA_VSTD::declval<_Tp&>()));

// [range.access.end]

_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__end)
  template<class _Tp>
  void end(_Tp&) = delete;
  template<class _Tp>
  void end(const _Tp&) = delete;

#if _CCCL_STD_VER > 2017
  template <class _Tp>
  concept __member_end =
    __can_borrow<_Tp> &&
    __workaround_52970<_Tp> &&
    requires(_Tp&& __t) {
      typename iterator_t<_Tp>;
      { _LIBCUDACXX_AUTO_CAST(__t.end()) } -> sentinel_for<iterator_t<_Tp>>;
    };

  template <class _Tp>
  concept __unqualified_end =
    !__member_end<_Tp> &&
    __can_borrow<_Tp> &&
    __class_or_enum<remove_cvref_t<_Tp>> &&
    requires(_Tp && __t) {
      typename iterator_t<_Tp>;
      { _LIBCUDACXX_AUTO_CAST(end(__t)) } -> sentinel_for<iterator_t<_Tp>>;
    };
#else // ^^^ CXX20 ^^^ / vvv CXX17 vvv
template<class _Tp>
  _LIBCUDACXX_CONCEPT_FRAGMENT(
    __member_end_,
    requires(_Tp&& __t)(
      requires(__can_borrow<_Tp>),
      requires(__workaround_52970<_Tp>),
      typename(iterator_t<_Tp>),
      requires(sentinel_for<decltype(_LIBCUDACXX_AUTO_CAST(__t.end())), iterator_t<_Tp>>)
    ));

  template<class _Tp>
  _LIBCUDACXX_CONCEPT __member_end = _LIBCUDACXX_FRAGMENT(__member_end_, _Tp);

  template<class _Tp>
  _LIBCUDACXX_CONCEPT_FRAGMENT(
    __unqualified_end_,
    requires(_Tp&& __t)(
      requires(!__member_end<_Tp>),
      requires(__can_borrow<_Tp>),
      requires(__class_or_enum<remove_cvref_t<_Tp>>),
      typename(iterator_t<_Tp>),
      requires(sentinel_for<decltype(_LIBCUDACXX_AUTO_CAST(end(__t))), iterator_t<_Tp>>)
    ));

  template<class _Tp>
  _LIBCUDACXX_CONCEPT __unqualified_end = _LIBCUDACXX_FRAGMENT(__unqualified_end_, _Tp);
#endif // _CCCL_STD_VER < 2020

  struct __fn {
    _LIBCUDACXX_TEMPLATE(class _Tp, size_t _Np)
      _LIBCUDACXX_REQUIRES((sizeof(_Tp) >= 0)) // Disallow incomplete element types.
    _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr auto operator()(_Tp (&__t)[_Np]) const noexcept
    {
      return __t + _Np;
    }

    _LIBCUDACXX_TEMPLATE(class _Tp)
      _LIBCUDACXX_REQUIRES(__member_end<_Tp>)
    _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr auto operator()(_Tp&& __t) const noexcept(noexcept(_LIBCUDACXX_AUTO_CAST(__t.end())))
    {
      return _LIBCUDACXX_AUTO_CAST(__t.end());
    }

    _LIBCUDACXX_TEMPLATE(class _Tp)
      _LIBCUDACXX_REQUIRES(__unqualified_end<_Tp>)
    _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr auto operator()(_Tp&& __t) const noexcept(noexcept(_LIBCUDACXX_AUTO_CAST(end(__t))))
    {
      return _LIBCUDACXX_AUTO_CAST(end(__t));
    }

    _LIBCUDACXX_TEMPLATE(class _Tp)
      _LIBCUDACXX_REQUIRES((!__member_end<_Tp>) _LIBCUDACXX_AND
                (!__unqualified_end<_Tp>))
    void operator()(_Tp&&) const = delete;
  };
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo {
  _LIBCUDACXX_CPO_ACCESSIBILITY auto end = __end::__fn{};
} // namespace __cpo

// [range.access.cbegin]

_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__cbegin)
  struct __fn {
    _LIBCUDACXX_TEMPLATE(class _Tp)
      _LIBCUDACXX_REQUIRES(is_lvalue_reference_v<_Tp&&>)
    _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr auto operator()(_Tp&& __t) const
      noexcept(noexcept(_CUDA_VRANGES::begin(static_cast<const remove_reference_t<_Tp>&>(__t))))
      -> decltype(      _CUDA_VRANGES::begin(static_cast<const remove_reference_t<_Tp>&>(__t)))
      { return          _CUDA_VRANGES::begin(static_cast<const remove_reference_t<_Tp>&>(__t)); }

    _LIBCUDACXX_TEMPLATE(class _Tp)
      _LIBCUDACXX_REQUIRES(is_rvalue_reference_v<_Tp&&>)
    _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr auto operator()(_Tp&& __t) const
      noexcept(noexcept(_CUDA_VRANGES::begin(static_cast<const _Tp&&>(__t))))
      -> decltype(      _CUDA_VRANGES::begin(static_cast<const _Tp&&>(__t)))
      { return          _CUDA_VRANGES::begin(static_cast<const _Tp&&>(__t)); }
  };
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo {
  _LIBCUDACXX_CPO_ACCESSIBILITY auto cbegin = __cbegin::__fn{};
} // namespace __cpo

// [range.access.cend]

_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__cend)
  struct __fn {
    _LIBCUDACXX_TEMPLATE(class _Tp)
      _LIBCUDACXX_REQUIRES(is_lvalue_reference_v<_Tp&&>)
    _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr auto operator()(_Tp&& __t) const
      noexcept(noexcept(_CUDA_VRANGES::end(static_cast<const remove_reference_t<_Tp>&>(__t))))
      -> decltype(      _CUDA_VRANGES::end(static_cast<const remove_reference_t<_Tp>&>(__t)))
      { return          _CUDA_VRANGES::end(static_cast<const remove_reference_t<_Tp>&>(__t)); }

    _LIBCUDACXX_TEMPLATE(class _Tp)
      _LIBCUDACXX_REQUIRES(is_rvalue_reference_v<_Tp&&>)
    _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr auto operator()(_Tp&& __t) const
      noexcept(noexcept(_CUDA_VRANGES::end(static_cast<const _Tp&&>(__t))))
      -> decltype(      _CUDA_VRANGES::end(static_cast<const _Tp&&>(__t)))
      { return          _CUDA_VRANGES::end(static_cast<const _Tp&&>(__t)); }
  };
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo {
  _LIBCUDACXX_CPO_ACCESSIBILITY auto cend = __cend::__fn{};
} // namespace __cpo
#endif // _CCCL_STD_VER > 2014 && !_CCCL_COMPILER_MSVC_2017

_LIBCUDACXX_END_NAMESPACE_RANGES

#endif // _LIBCUDACXX___RANGES_ACCESS_H
