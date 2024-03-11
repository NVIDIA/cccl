// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_REVERSE_ITERATOR_H
#define _LIBCUDACXX___ITERATOR_REVERSE_ITERATOR_H

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

#include "../__algorithm/unwrap_iter.h"
#ifndef _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR
#  include "../__compare/compare_three_way_result.h"
#  include "../__compare/three_way_comparable.h"
#endif // _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR
#include "../__concepts/convertible_to.h"
#include "../__iterator/advance.h"
#include "../__iterator/concepts.h"
#include "../__iterator/incrementable_traits.h"
#include "../__iterator/iter_move.h"
#include "../__iterator/iter_swap.h"
#include "../__iterator/iterator.h"
#include "../__iterator/iterator_traits.h"
#include "../__iterator/next.h"
#include "../__iterator/prev.h"
#include "../__iterator/readable_traits.h"
#include "../__memory/addressof.h"
#ifdef _LIBCUDACXX_HAS_RANGES
#  include "../__ranges/access.h"
#  include "../__ranges/concepts.h"
#  include "../__ranges/subrange.h"
#endif // _LIBCUDACXX_HAS_RANGES
#include "../__type_traits/conditional.h"
#include "../__type_traits/enable_if.h"
#include "../__type_traits/is_assignable.h"
#include "../__type_traits/is_convertible.h"
#include "../__type_traits/is_nothrow_copy_constructible.h"
#include "../__type_traits/is_same.h"
#include "../__type_traits/void_t.h"
#include "../__utility/declval.h"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_STD_VER >= 2017
template <class _Iter, class = void>
_LIBCUDACXX_INLINE_VAR constexpr bool __noexcept_rev_iter_iter_move = false;

template <class _Iter>
_LIBCUDACXX_INLINE_VAR constexpr bool __noexcept_rev_iter_iter_move<_Iter, void_t<decltype(--_CUDA_VSTD::declval<_Iter&>())>> =
  is_nothrow_copy_constructible_v<_Iter>&& noexcept(_CUDA_VRANGES::iter_move(--_CUDA_VSTD::declval<_Iter&>()));

template <class _Iter, class _Iter2, class = void>
_LIBCUDACXX_INLINE_VAR constexpr bool __noexcept_rev_iter_iter_swap = false;

template <class _Iter, class _Iter2>
_LIBCUDACXX_INLINE_VAR constexpr bool __noexcept_rev_iter_iter_swap<_Iter, _Iter2,
  enable_if_t<indirectly_swappable<_Iter, _Iter2>>> =
  is_nothrow_copy_constructible_v<_Iter> && is_nothrow_copy_constructible_v<_Iter2> &&
  noexcept(_CUDA_VRANGES::iter_swap(--declval<_Iter&>(), --declval<_Iter2&>()));
#endif // _CCCL_STD_VER >= 2017

_CCCL_SUPPRESS_DEPRECATED_PUSH
template <class _Iter>
class _LIBCUDACXX_TEMPLATE_VIS reverse_iterator
#if _CCCL_STD_VER <= 2014 || !defined(_LIBCUDACXX_ABI_NO_ITERATOR_BASES)
    : public iterator<typename iterator_traits<_Iter>::iterator_category,
                      typename iterator_traits<_Iter>::value_type,
                      typename iterator_traits<_Iter>::difference_type,
                      typename iterator_traits<_Iter>::pointer,
                      typename iterator_traits<_Iter>::reference>
#endif
{
  _CCCL_SUPPRESS_DEPRECATED_POP

private:
#ifndef _LIBCUDACXX_ABI_NO_ITERATOR_BASES
  _Iter __t_; // no longer used as of LWG #2360, not removed due to ABI break
#endif

#if _CCCL_STD_VER > 2017
  static_assert(__is_cpp17_bidirectional_iterator<_Iter>::value || bidirectional_iterator<_Iter>,
                "reverse_iterator<It> requires It to be a bidirectional iterator.");
#endif // _CCCL_STD_VER > 2017

protected:
  _Iter current;

public:
  using iterator_type = _Iter;

  using iterator_category =
    _If<__is_cpp17_random_access_iterator<_Iter>::value,
        random_access_iterator_tag,
        typename iterator_traits<_Iter>::iterator_category>;
  using pointer = typename iterator_traits<_Iter>::pointer;
#if _CCCL_STD_VER > 2014
  using iterator_concept = _If<random_access_iterator<_Iter>, random_access_iterator_tag, bidirectional_iterator_tag>;
  using value_type       = iter_value_t<_Iter>;
  using difference_type  = iter_difference_t<_Iter>;
  using reference        = iter_reference_t<_Iter>;
#else
  using value_type      = typename iterator_traits<_Iter>::value_type;
  using difference_type = typename iterator_traits<_Iter>::difference_type;
  using reference       = typename iterator_traits<_Iter>::reference;
#endif

#ifndef _LIBCUDACXX_ABI_NO_ITERATOR_BASES
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 reverse_iterator()
      : __t_()
      , current()
  {}

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 explicit reverse_iterator(
    _Iter __x)
      : __t_(__x)
      , current(__x)
  {}

  template <class _Up, class = __enable_if_t< !is_same<_Up, _Iter>::value && is_convertible<_Up const&, _Iter>::value > >
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
  reverse_iterator(const reverse_iterator<_Up>& __u)
      : __t_(__u.base())
      , current(__u.base())
  {}

  template <class _Up,
            class = __enable_if_t< !is_same<_Up, _Iter>::value && is_convertible<_Up const&, _Iter>::value
                                   && is_assignable<_Iter&, _Up const&>::value > >
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 reverse_iterator&
  operator=(const reverse_iterator<_Up>& __u)
  {
    __t_ = current = __u.base();
    return *this;
  }
#else
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 reverse_iterator()
      : current()
  {}

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 explicit reverse_iterator(
    _Iter __x)
      : current(__x)
  {}

  template <class _Up, class = __enable_if_t< !is_same<_Up, _Iter>::value && is_convertible<_Up const&, _Iter>::value > >
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
  reverse_iterator(const reverse_iterator<_Up>& __u)
      : current(__u.base())
  {}

  template <class _Up,
            class = __enable_if_t< !is_same<_Up, _Iter>::value && is_convertible<_Up const&, _Iter>::value
                                   && is_assignable<_Iter&, _Up const&>::value > >
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 reverse_iterator&
  operator=(const reverse_iterator<_Up>& __u)
  {
    current = __u.base();
    return *this;
  }
#endif
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 _Iter base() const
  {
    return current;
  }
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 reference operator*() const
  {
    _Iter __tmp = current;
    return *--__tmp;
  }

#if _CCCL_STD_VER > 2017
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr pointer operator->() const
    requires is_pointer_v<_Iter> || requires(const _Iter __i) { __i.operator->(); }
  {
    if constexpr (is_pointer_v<_Iter>)
    {
      return _CUDA_VSTD::prev(current);
    }
    else
    {
      return _CUDA_VSTD::prev(current).operator->();
    }
  }
#else
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 pointer operator->() const
  {
    return _CUDA_VSTD::addressof(operator*());
  }
#endif // _CCCL_STD_VER > 2017

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 reverse_iterator&
  operator++()
  {
    --current;
    return *this;
  }
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 reverse_iterator
  operator++(int)
  {
    reverse_iterator __tmp(*this);
    --current;
    return __tmp;
  }
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 reverse_iterator&
  operator--()
  {
    ++current;
    return *this;
  }
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 reverse_iterator
  operator--(int)
  {
    reverse_iterator __tmp(*this);
    ++current;
    return __tmp;
  }
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 reverse_iterator
  operator+(difference_type __n) const
  {
    return reverse_iterator(current - __n);
  }
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 reverse_iterator&
  operator+=(difference_type __n)
  {
    current -= __n;
    return *this;
  }
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 reverse_iterator
  operator-(difference_type __n) const
  {
    return reverse_iterator(current + __n);
  }
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 reverse_iterator&
  operator-=(difference_type __n)
  {
    current += __n;
    return *this;
  }
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 reference
  operator[](difference_type __n) const
  {
    return *(*this + __n);
  }

#if _CCCL_STD_VER > 2014
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY friend constexpr iter_rvalue_reference_t<_Iter>
  iter_move(const reverse_iterator& __i) noexcept(__noexcept_rev_iter_iter_move<_Iter>)
  {
    auto __tmp = __i.base();
    return _CUDA_VRANGES::iter_move(--__tmp);
  }

#if defined(_CCCL_COMPILER_MSVC_2017) // MSVC2017 cannot find _Iter otherwise
  template<class _Iter2, class _Iter1 = _Iter>
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY friend constexpr auto
  iter_swap(const reverse_iterator<_Iter1>& __x, const reverse_iterator<_Iter2>& __y)
    noexcept(__noexcept_rev_iter_iter_swap<_Iter1, _Iter2>)
    _LIBCUDACXX_TRAILING_REQUIRES(void)(same_as<_Iter1, _Iter> && indirectly_swappable<_Iter2, _Iter1>)
  {
    auto __xtmp = __x.base();
    auto __ytmp = __y.base();
    _CUDA_VRANGES::iter_swap(--__xtmp, --__ytmp);
  }
#else // ^^^ _CCCL_COMPILER_MSVC_2017 ^^^ / vvv !_CCCL_COMPILER_MSVC_2017 vvv
  template<class _Iter2>
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY friend constexpr auto
  iter_swap(const reverse_iterator& __x, const reverse_iterator<_Iter2>& __y)
    noexcept(__noexcept_rev_iter_iter_swap<_Iter, _Iter2>)
    _LIBCUDACXX_TRAILING_REQUIRES(void)(indirectly_swappable<_Iter2, _Iter>)
  {
    auto __xtmp = __x.base();
    auto __ytmp = __y.base();
    _CUDA_VRANGES::iter_swap(--__xtmp, --__ytmp);
#if defined(_CCCL_COMPILER_ICC)
  _LIBCUDACXX_UNREACHABLE();
#endif // _CCCL_COMPILER_ICC
  }
#endif // !_CCCL_COMPILER_MSVC_2017
#endif // _CCCL_STD_VER > 2014
};

template <class _Iter>
struct __is_reverse_iterator : false_type
{};

template <class _Iter>
struct __is_reverse_iterator<reverse_iterator<_Iter> > : true_type
{};

template <class _Iter1, class _Iter2>
inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 bool
operator==(const reverse_iterator<_Iter1>& __x, const reverse_iterator<_Iter2>& __y)
#if _CCCL_STD_VER > 2017
  requires requires {
    {
      __x.base() == __y.base()
    } -> convertible_to<bool>;
  }
#endif // _CCCL_STD_VER > 2017
{
  return __x.base() == __y.base();
}

template <class _Iter1, class _Iter2>
inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 bool
operator<(const reverse_iterator<_Iter1>& __x, const reverse_iterator<_Iter2>& __y)
#if _CCCL_STD_VER > 2017
  requires requires {
    {
      __x.base() > __y.base()
    } -> convertible_to<bool>;
  }
#endif // _CCCL_STD_VER > 2017
{
  return __x.base() > __y.base();
}

template <class _Iter1, class _Iter2>
inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 bool
operator!=(const reverse_iterator<_Iter1>& __x, const reverse_iterator<_Iter2>& __y)
#if _CCCL_STD_VER > 2017
  requires requires {
    {
      __x.base() != __y.base()
    } -> convertible_to<bool>;
  }
#endif // _CCCL_STD_VER > 2017
{
  return __x.base() != __y.base();
}

template <class _Iter1, class _Iter2>
inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 bool
operator>(const reverse_iterator<_Iter1>& __x, const reverse_iterator<_Iter2>& __y)
#if _CCCL_STD_VER > 2017
  requires requires {
    {
      __x.base() < __y.base()
    } -> convertible_to<bool>;
  }
#endif // _CCCL_STD_VER > 2017
{
  return __x.base() < __y.base();
}

template <class _Iter1, class _Iter2>
inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 bool
operator>=(const reverse_iterator<_Iter1>& __x, const reverse_iterator<_Iter2>& __y)
#if _CCCL_STD_VER > 2017
  requires requires {
    {
      __x.base() <= __y.base()
    } -> convertible_to<bool>;
  }
#endif // _CCCL_STD_VER > 2017
{
  return __x.base() <= __y.base();
}

template <class _Iter1, class _Iter2>
inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 bool
operator<=(const reverse_iterator<_Iter1>& __x, const reverse_iterator<_Iter2>& __y)
#if _CCCL_STD_VER > 2017
  requires requires {
    {
      __x.base() >= __y.base()
    } -> convertible_to<bool>;
  }
#endif // _CCCL_STD_VER > 2017
{
  return __x.base() >= __y.base();
}

#ifndef _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR
template <class _Iter1, three_way_comparable_with<_Iter1> _Iter2>
_LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr compare_three_way_result_t<_Iter1, _Iter2>
operator<=>(const reverse_iterator<_Iter1>& __x, const reverse_iterator<_Iter2>& __y)
{
  return __y.base() <=> __x.base();
}
#endif // _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR

template <class _Iter1, class _Iter2>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 auto
operator-(const reverse_iterator<_Iter1>& __x, const reverse_iterator<_Iter2>& __y) -> decltype(__y.base() - __x.base())
{
  return __y.base() - __x.base();
}

template <class _Iter>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 reverse_iterator<_Iter>
operator+(typename reverse_iterator<_Iter>::difference_type __n, const reverse_iterator<_Iter>& __x)
{
  return reverse_iterator<_Iter>(__x.base() - __n);
}

#if _CCCL_STD_VER > 2017
template <class _Iter1, class _Iter2>
  requires(!sized_sentinel_for<_Iter1, _Iter2>)
inline constexpr bool disable_sized_sentinel_for<reverse_iterator<_Iter1>, reverse_iterator<_Iter2>> = true;
#endif // _CCCL_STD_VER > 2017

#if _CCCL_STD_VER > 2011
template <class _Iter>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 reverse_iterator<_Iter>
make_reverse_iterator(_Iter __i)
{
  return reverse_iterator<_Iter>(__i);
}
#endif

#if _CCCL_STD_VER <= 2017
template <class _Iter>
using __unconstrained_reverse_iterator = reverse_iterator<_Iter>;
#else

// __unconstrained_reverse_iterator allows us to use reverse iterators in the implementation of algorithms by working
// around a language issue in C++20.
// In C++20, when a reverse iterator wraps certain C++20-hostile iterators, calling comparison operators on it will
// result in a compilation error. However, calling comparison operators on the pristine hostile iterator is not
// an error. Thus, we cannot use reverse_iterators in the implementation of an algorithm that accepts a
// C++20-hostile iterator. This class is an internal workaround -- it is a copy of reverse_iterator with
// tweaks to make it support hostile iterators.
//
// A C++20-hostile iterator is one that defines a comparison operator where one of the arguments is an exact match
// and the other requires an implicit conversion, for example:
//   friend bool operator==(const BaseIter&, const DerivedIter&);
//
// C++20 rules for rewriting equality operators create another overload of this function with parameters reversed:
//   friend bool operator==(const DerivedIter&, const BaseIter&);
//
// This creates an ambiguity in overload resolution.
//
// Clang treats this ambiguity differently in different contexts. When operator== is actually called in the function
// body, the code is accepted with a warning. When a concept requires operator== to be a valid expression, however,
// it evaluates to false. Thus, the implementation of reverse_iterator::operator== can actually call operator== on its
// base iterators, but the constraints on reverse_iterator::operator== prevent it from being considered during overload
// resolution. This class simply removes the problematic constraints from comparison functions.
template <class _Iter>
class __unconstrained_reverse_iterator
{
  _Iter __iter_;

public:
  static_assert(__is_cpp17_bidirectional_iterator<_Iter>::value || bidirectional_iterator<_Iter>);

  using iterator_type = _Iter;
  using iterator_category =
    _If<__is_cpp17_random_access_iterator<_Iter>::value, random_access_iterator_tag, __iterator_category_type<_Iter>>;
  using pointer         = __iterator_pointer_type<_Iter>;
  using value_type      = iter_value_t<_Iter>;
  using difference_type = iter_difference_t<_Iter>;
  using reference       = iter_reference_t<_Iter>;

  _LIBCUDACXX_HIDE_FROM_ABI constexpr __unconstrained_reverse_iterator() = default;
  _LIBCUDACXX_HIDE_FROM_ABI constexpr __unconstrained_reverse_iterator(const __unconstrained_reverse_iterator&) =
    default;
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr explicit __unconstrained_reverse_iterator(
    _Iter __iter)
      : __iter_(__iter)
  {}

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr _Iter base() const
  {
    return __iter_;
  }
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr reference operator*() const
  {
    auto __tmp = __iter_;
    return *--__tmp;
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr pointer operator->() const
  {
    if constexpr (is_pointer_v<_Iter>)
    {
      return _CUDA_VSTD::prev(__iter_);
    }
    else
    {
      return _CUDA_VSTD::prev(__iter_).operator->();
    }
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY friend constexpr iter_rvalue_reference_t<_Iter>
  iter_move(const __unconstrained_reverse_iterator& __i) noexcept(
    is_nothrow_copy_constructible_v<_Iter>&& noexcept(_CUDA_VRANGES::iter_move(--declval<_Iter&>())))
  {
    auto __tmp = __i.base();
    return _CUDA_VRANGES::iter_move(--__tmp);
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr __unconstrained_reverse_iterator& operator++()
  {
    --__iter_;
    return *this;
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr __unconstrained_reverse_iterator operator++(int)
  {
    auto __tmp = *this;
    --__iter_;
    return __tmp;
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr __unconstrained_reverse_iterator& operator--()
  {
    ++__iter_;
    return *this;
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr __unconstrained_reverse_iterator operator--(int)
  {
    auto __tmp = *this;
    ++__iter_;
    return __tmp;
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr __unconstrained_reverse_iterator&
  operator+=(difference_type __n)
  {
    __iter_ -= __n;
    return *this;
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr __unconstrained_reverse_iterator&
  operator-=(difference_type __n)
  {
    __iter_ += __n;
    return *this;
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr __unconstrained_reverse_iterator
  operator+(difference_type __n) const
  {
    return __unconstrained_reverse_iterator(__iter_ - __n);
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr __unconstrained_reverse_iterator
  operator-(difference_type __n) const
  {
    return __unconstrained_reverse_iterator(__iter_ + __n);
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr difference_type
  operator-(const __unconstrained_reverse_iterator& __other) const
  {
    return __other.__iter_ - __iter_;
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr auto operator[](difference_type __n) const
  {
    return *(*this + __n);
  }

  // Deliberately unconstrained unlike the comparison functions in `reverse_iterator` -- see the class comment for the
  // rationale.
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY friend constexpr bool
  operator==(const __unconstrained_reverse_iterator& __lhs, const __unconstrained_reverse_iterator& __rhs)
  {
    return __lhs.base() == __rhs.base();
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY friend constexpr bool
  operator!=(const __unconstrained_reverse_iterator& __lhs, const __unconstrained_reverse_iterator& __rhs)
  {
    return __lhs.base() != __rhs.base();
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY friend constexpr bool
  operator<(const __unconstrained_reverse_iterator& __lhs, const __unconstrained_reverse_iterator& __rhs)
  {
    return __lhs.base() > __rhs.base();
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY friend constexpr bool
  operator>(const __unconstrained_reverse_iterator& __lhs, const __unconstrained_reverse_iterator& __rhs)
  {
    return __lhs.base() < __rhs.base();
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY friend constexpr bool
  operator<=(const __unconstrained_reverse_iterator& __lhs, const __unconstrained_reverse_iterator& __rhs)
  {
    return __lhs.base() >= __rhs.base();
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY friend constexpr bool
  operator>=(const __unconstrained_reverse_iterator& __lhs, const __unconstrained_reverse_iterator& __rhs)
  {
    return __lhs.base() <= __rhs.base();
  }
};

template <class _Iter>
struct __is_reverse_iterator<__unconstrained_reverse_iterator<_Iter>> : true_type
{};

#endif // _CCCL_STD_VER <= 2017

template <template <class> class _RevIter1, template <class> class _RevIter2, class _Iter>
struct __unwrap_reverse_iter_impl
{
  using _UnwrappedIter  = decltype(__unwrap_iter_impl<_Iter>::__unwrap(_CUDA_VSTD::declval<_Iter>()));
  using _ReverseWrapper = _RevIter1<_RevIter2<_Iter> >;

  static _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr _ReverseWrapper
  __rewrap(_ReverseWrapper __orig_iter, _UnwrappedIter __unwrapped_iter)
  {
    return _ReverseWrapper(
      _RevIter2<_Iter>(__unwrap_iter_impl<_Iter>::__rewrap(__orig_iter.base().base(), __unwrapped_iter)));
  }

  static _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr _UnwrappedIter
  __unwrap(_ReverseWrapper __i) noexcept
  {
    return __unwrap_iter_impl<_Iter>::__unwrap(__i.base().base());
  }
};

#ifdef _LIBCUDACXX_HAS_RANGES
#  if _CCCL_STD_VER > 2014
template <_CUDA_VRANGES::bidirectional_range _Range>
_LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr _CUDA_VRANGES::
  subrange<reverse_iterator<_CUDA_VRANGES::iterator_t<_Range>>, reverse_iterator<_CUDA_VRANGES::iterator_t<_Range>>>
  __reverse_range(_Range&& __range)
{
  auto __first = _CUDA_VRANGES::begin(__range);
  return {_CUDA_VSTD::make_reverse_iterator(_CUDA_VRANGES::next(__first, _CUDA_VRANGES::end(__range))),
          _CUDA_VSTD::make_reverse_iterator(__first)};
}
#  endif // _CCCL_STD_VER > 2014
#endif // _LIBCUDACXX_HAS_RANGES

template <class _Iter, bool __b>
struct __unwrap_iter_impl<reverse_iterator<reverse_iterator<_Iter> >, __b>
    : __unwrap_reverse_iter_impl<reverse_iterator, reverse_iterator, _Iter>
{};

#if _CCCL_STD_VER > 2017

template <class _Iter, bool __b>
struct __unwrap_iter_impl<reverse_iterator<__unconstrained_reverse_iterator<_Iter>>, __b>
    : __unwrap_reverse_iter_impl<reverse_iterator, __unconstrained_reverse_iterator, _Iter>
{};

template <class _Iter, bool __b>
struct __unwrap_iter_impl<__unconstrained_reverse_iterator<reverse_iterator<_Iter>>, __b>
    : __unwrap_reverse_iter_impl<__unconstrained_reverse_iterator, reverse_iterator, _Iter>
{};

template <class _Iter, bool __b>
struct __unwrap_iter_impl<__unconstrained_reverse_iterator<__unconstrained_reverse_iterator<_Iter>>, __b>
    : __unwrap_reverse_iter_impl<__unconstrained_reverse_iterator, __unconstrained_reverse_iterator, _Iter>
{};

#endif // _CCCL_STD_VER > 2017

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ITERATOR_REVERSE_ITERATOR_H
