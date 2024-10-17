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

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__memory/pointer_traits.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_trivially_copy_assignable.h>

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
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 __wrap_iter() noexcept
      : __i_()
  {}
  template <class _Up>
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14
  __wrap_iter(const __wrap_iter<_Up>& __u,
              typename enable_if<is_convertible<_Up, iterator_type>::value>::type* = nullptr) noexcept
      : __i_(__u.base())
  {}
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 reference operator*() const noexcept
  {
    return *__i_;
  }
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 pointer operator->() const noexcept
  {
    return _CUDA_VSTD::__to_address(__i_);
  }
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 __wrap_iter& operator++() noexcept
  {
    ++__i_;
    return *this;
  }
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 __wrap_iter operator++(int) noexcept
  {
    __wrap_iter __tmp(*this);
    ++(*this);
    return __tmp;
  }

  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 __wrap_iter& operator--() noexcept
  {
    --__i_;
    return *this;
  }
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 __wrap_iter operator--(int) noexcept
  {
    __wrap_iter __tmp(*this);
    --(*this);
    return __tmp;
  }
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 __wrap_iter operator+(difference_type __n) const noexcept
  {
    __wrap_iter __w(*this);
    __w += __n;
    return __w;
  }
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 __wrap_iter& operator+=(difference_type __n) noexcept
  {
    __i_ += __n;
    return *this;
  }
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 __wrap_iter operator-(difference_type __n) const noexcept
  {
    return *this + (-__n);
  }
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 __wrap_iter& operator-=(difference_type __n) noexcept
  {
    *this += -__n;
    return *this;
  }
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 reference operator[](difference_type __n) const noexcept
  {
    return __i_[__n];
  }

  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 iterator_type base() const noexcept
  {
    return __i_;
  }

private:
  _LIBCUDACXX_HIDE_FROM_ABI constexpr __wrap_iter(iterator_type __x) noexcept
      : __i_(__x)
  {}

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
_LIBCUDACXX_HIDE_FROM_ABI constexpr bool
operator==(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter1>& __y) noexcept
{
  return __x.base() == __y.base();
}

template <class _Iter1, class _Iter2>
_LIBCUDACXX_HIDE_FROM_ABI constexpr bool
operator==(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter2>& __y) noexcept
{
  return __x.base() == __y.base();
}

template <class _Iter1>
_LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 bool
operator<(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter1>& __y) noexcept
{
  return __x.base() < __y.base();
}

template <class _Iter1, class _Iter2>
_LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 bool
operator<(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter2>& __y) noexcept
{
  return __x.base() < __y.base();
}

template <class _Iter1>
_LIBCUDACXX_HIDE_FROM_ABI constexpr bool
operator!=(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter1>& __y) noexcept
{
  return !(__x == __y);
}

template <class _Iter1, class _Iter2>
_LIBCUDACXX_HIDE_FROM_ABI constexpr bool
operator!=(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter2>& __y) noexcept
{
  return !(__x == __y);
}

template <class _Iter1>
_LIBCUDACXX_HIDE_FROM_ABI constexpr bool
operator>(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter1>& __y) noexcept
{
  return __y < __x;
}

template <class _Iter1, class _Iter2>
_LIBCUDACXX_HIDE_FROM_ABI constexpr bool
operator>(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter2>& __y) noexcept
{
  return __y < __x;
}

template <class _Iter1>
_LIBCUDACXX_HIDE_FROM_ABI constexpr bool
operator>=(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter1>& __y) noexcept
{
  return !(__x < __y);
}

template <class _Iter1, class _Iter2>
_LIBCUDACXX_HIDE_FROM_ABI constexpr bool
operator>=(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter2>& __y) noexcept
{
  return !(__x < __y);
}

template <class _Iter1>
_LIBCUDACXX_HIDE_FROM_ABI constexpr bool
operator<=(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter1>& __y) noexcept
{
  return !(__y < __x);
}

template <class _Iter1, class _Iter2>
_LIBCUDACXX_HIDE_FROM_ABI constexpr bool
operator<=(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter2>& __y) noexcept
{
  return !(__y < __x);
}

template <class _Iter1, class _Iter2>
_LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 auto
operator-(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter2>& __y) noexcept -> decltype(__x.base() - __y.base())
{
  return __x.base() - __y.base();
}

template <class _Iter1>
_LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 __wrap_iter<_Iter1>
operator+(typename __wrap_iter<_Iter1>::difference_type __n, __wrap_iter<_Iter1> __x) noexcept
{
  __x += __n;
  return __x;
}

#if _CCCL_STD_VER <= 2017
template <class _It>
struct __is_cpp17_contiguous_iterator<__wrap_iter<_It>> : true_type
{};
#endif

template <class _It>
struct _CCCL_TYPE_VISIBILITY_DEFAULT pointer_traits<__wrap_iter<_It>>
{
  typedef __wrap_iter<_It> pointer;
  typedef typename pointer_traits<_It>::element_type element_type;
  typedef typename pointer_traits<_It>::difference_type difference_type;

  _LIBCUDACXX_HIDE_FROM_ABI constexpr static element_type* to_address(pointer __w) noexcept
  {
    return _CUDA_VSTD::__to_address(__w.base());
  }
};

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ITERATOR_WRAP_ITER_H
