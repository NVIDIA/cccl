// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_WRAP_ITER_H
#define _LIBCUDACXX___ITERATOR_WRAP_ITER_H

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

#include "../__debug"
#include "../__iterator/iterator_traits.h"
#include "../__memory/addressof.h"
#include "../__memory/pointer_traits.h"
#include "../__type_traits/enable_if.h"
#include "../__type_traits/is_convertible.h"
#include "../__type_traits/is_trivially_copy_assignable.h"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Iter>
class __wrap_iter
{
public:
  typedef _Iter iterator_type;
  typedef typename iterator_traits<iterator_type>::value_type value_type;
  typedef typename iterator_traits<iterator_type>::difference_type difference_type;
  typedef typename iterator_traits<iterator_type>::pointer pointer;
  typedef typename iterator_traits<iterator_type>::reference reference;
  typedef typename iterator_traits<iterator_type>::iterator_category iterator_category;
#if _CCCL_STD_VER > 2011
  typedef contiguous_iterator_tag iterator_concept;
#endif

private:
  iterator_type __i_;

public:
  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 __wrap_iter() noexcept
      : __i_()
  {
    _CUDA_VSTD::__debug_db_insert_i(this);
  }
  template <class _Up>
  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
  __wrap_iter(const __wrap_iter<_Up>& __u,
              typename enable_if<is_convertible<_Up, iterator_type>::value>::type* = nullptr) noexcept
      : __i_(__u.base())
  {
#ifdef _LIBCUDACXX_ENABLE_DEBUG_MODE
    if (!__libcpp_is_constant_evaluated())
    {
      __get_db()->__iterator_copy(this, _CUDA_VSTD::addressof(__u));
    }
#endif
  }
#ifdef _LIBCUDACXX_ENABLE_DEBUG_MODE
  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 __wrap_iter(const __wrap_iter& __x)
      : __i_(__x.base())
  {
    if (!__libcpp_is_constant_evaluated())
    {
      __get_db()->__iterator_copy(this, _CUDA_VSTD::addressof(__x));
    }
  }
  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 __wrap_iter& operator=(const __wrap_iter& __x)
  {
    if (this != _CUDA_VSTD::addressof(__x))
    {
      if (!__libcpp_is_constant_evaluated())
      {
        __get_db()->__iterator_copy(this, _CUDA_VSTD::addressof(__x));
      }
      __i_ = __x.__i_;
    }
    return *this;
  }
  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17 ~__wrap_iter()
  {
    if (!__libcpp_is_constant_evaluated())
    {
      __get_db()->__erase_i(this);
    }
  }
#endif
  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 reference operator*() const noexcept
  {
    _LIBCUDACXX_DEBUG_ASSERT(__get_const_db()->__dereferenceable(this),
                             "Attempted to dereference a non-dereferenceable iterator");
    return *__i_;
  }
  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 pointer operator->() const noexcept
  {
    _LIBCUDACXX_DEBUG_ASSERT(__get_const_db()->__dereferenceable(this),
                             "Attempted to dereference a non-dereferenceable iterator");
    return _CUDA_VSTD::__to_address(__i_);
  }
  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 __wrap_iter& operator++() noexcept
  {
    _LIBCUDACXX_DEBUG_ASSERT(__get_const_db()->__dereferenceable(this),
                             "Attempted to increment a non-incrementable iterator");
    ++__i_;
    return *this;
  }
  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 __wrap_iter operator++(int) noexcept
  {
    __wrap_iter __tmp(*this);
    ++(*this);
    return __tmp;
  }

  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 __wrap_iter& operator--() noexcept
  {
    _LIBCUDACXX_DEBUG_ASSERT(__get_const_db()->__decrementable(this),
                             "Attempted to decrement a non-decrementable iterator");
    --__i_;
    return *this;
  }
  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 __wrap_iter operator--(int) noexcept
  {
    __wrap_iter __tmp(*this);
    --(*this);
    return __tmp;
  }
  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 __wrap_iter
  operator+(difference_type __n) const noexcept
  {
    __wrap_iter __w(*this);
    __w += __n;
    return __w;
  }
  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 __wrap_iter& operator+=(difference_type __n) noexcept
  {
    _LIBCUDACXX_DEBUG_ASSERT(__get_const_db()->__addable(this, __n),
                             "Attempted to add/subtract an iterator outside its valid range");
    __i_ += __n;
    return *this;
  }
  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 __wrap_iter
  operator-(difference_type __n) const noexcept
  {
    return *this + (-__n);
  }
  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 __wrap_iter& operator-=(difference_type __n) noexcept
  {
    *this += -__n;
    return *this;
  }
  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 reference
  operator[](difference_type __n) const noexcept
  {
    _LIBCUDACXX_DEBUG_ASSERT(__get_const_db()->__subscriptable(this, __n),
                             "Attempted to subscript an iterator outside its valid range");
    return __i_[__n];
  }

  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 iterator_type base() const noexcept
  {
    return __i_;
  }

// private:
#if _LIBCUDACXX_DEBUG_LEVEL >= 2
  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_IF_NODEBUG __wrap_iter(const void* __p, iterator_type __x)
      : __i_(__x)
  {
    if (!__libcpp_is_constant_evaluated())
    {
      __get_db()->__insert_ic(this, __p);
    }
  }
#else
  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_IF_NODEBUG __wrap_iter(iterator_type __x) noexcept
      : __i_(__x)
  {}
#endif

  template <class _Up>
  friend class __wrap_iter;
  template <class _CharT, class _Traits, class _Alloc>
  friend class basic_string;
  template <class _Tp, class _Alloc>
  friend class vector;
  template <class _Tp, size_t>
  friend class span;
};

template <class _Iter1>
_LIBCUDACXX_INLINE_VISIBILITY constexpr bool
operator==(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter1>& __y) noexcept
{
  return __x.base() == __y.base();
}

template <class _Iter1, class _Iter2>
_LIBCUDACXX_INLINE_VISIBILITY constexpr bool
operator==(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter2>& __y) noexcept
{
  return __x.base() == __y.base();
}

template <class _Iter1>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 bool
operator<(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter1>& __y) noexcept
{
  _LIBCUDACXX_DEBUG_ASSERT(
    __get_const_db()->__less_than_comparable(_CUDA_VSTD::addressof(__x), _CUDA_VSTD::addressof(__y)),
    "Attempted to compare incomparable iterators");
  return __x.base() < __y.base();
}

template <class _Iter1, class _Iter2>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 bool
operator<(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter2>& __y) noexcept
{
  _LIBCUDACXX_DEBUG_ASSERT(__get_const_db()->__less_than_comparable(&__x, &__y),
                           "Attempted to compare incomparable iterators");
  return __x.base() < __y.base();
}

template <class _Iter1>
_LIBCUDACXX_INLINE_VISIBILITY constexpr bool
operator!=(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter1>& __y) noexcept
{
  return !(__x == __y);
}

template <class _Iter1, class _Iter2>
_LIBCUDACXX_INLINE_VISIBILITY constexpr bool
operator!=(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter2>& __y) noexcept
{
  return !(__x == __y);
}

template <class _Iter1>
_LIBCUDACXX_INLINE_VISIBILITY constexpr bool
operator>(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter1>& __y) noexcept
{
  return __y < __x;
}

template <class _Iter1, class _Iter2>
_LIBCUDACXX_INLINE_VISIBILITY constexpr bool
operator>(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter2>& __y) noexcept
{
  return __y < __x;
}

template <class _Iter1>
_LIBCUDACXX_INLINE_VISIBILITY constexpr bool
operator>=(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter1>& __y) noexcept
{
  return !(__x < __y);
}

template <class _Iter1, class _Iter2>
_LIBCUDACXX_INLINE_VISIBILITY constexpr bool
operator>=(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter2>& __y) noexcept
{
  return !(__x < __y);
}

template <class _Iter1>
_LIBCUDACXX_INLINE_VISIBILITY constexpr bool
operator<=(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter1>& __y) noexcept
{
  return !(__y < __x);
}

template <class _Iter1, class _Iter2>
_LIBCUDACXX_INLINE_VISIBILITY constexpr bool
operator<=(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter2>& __y) noexcept
{
  return !(__y < __x);
}

template <class _Iter1, class _Iter2>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 auto
operator-(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter2>& __y) noexcept -> decltype(__x.base() - __y.base())
{
  _LIBCUDACXX_DEBUG_ASSERT(
    __get_const_db()->__less_than_comparable(_CUDA_VSTD::addressof(__x), _CUDA_VSTD::addressof(__y)),
    "Attempted to subtract incompatible iterators");
  return __x.base() - __y.base();
}

template <class _Iter1>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 __wrap_iter<_Iter1>
operator+(typename __wrap_iter<_Iter1>::difference_type __n, __wrap_iter<_Iter1> __x) noexcept
{
  __x += __n;
  return __x;
}

#if _CCCL_STD_VER <= 2017
template <class _It>
struct __is_cpp17_contiguous_iterator<__wrap_iter<_It> > : true_type
{};
#endif

template <class _It>
struct _LIBCUDACXX_TEMPLATE_VIS pointer_traits<__wrap_iter<_It> >
{
  typedef __wrap_iter<_It> pointer;
  typedef typename pointer_traits<_It>::element_type element_type;
  typedef typename pointer_traits<_It>::difference_type difference_type;

  _LIBCUDACXX_INLINE_VISIBILITY constexpr static element_type* to_address(pointer __w) noexcept
  {
    return _CUDA_VSTD::__to_address(__w.base());
  }
};

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ITERATOR_WRAP_ITER_H
