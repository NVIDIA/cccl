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

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/unwrap_iter.h>
#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#  include <cuda/std/__compare/compare_three_way_result.h>
#  include <cuda/std/__compare/three_way_comparable.h>
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#include <cuda/std/__concepts/convertible_to.h>
#include <cuda/std/__iterator/advance.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/incrementable_traits.h>
#include <cuda/std/__iterator/iter_move.h>
#include <cuda/std/__iterator/iter_swap.h>
#include <cuda/std/__iterator/iterator.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__iterator/next.h>
#include <cuda/std/__iterator/prev.h>
#include <cuda/std/__iterator/readable_traits.h>
#include <cuda/std/__memory/addressof.h>
#ifdef _LIBCUDACXX_HAS_RANGES
#  include <cuda/std/__ranges/access.h>
#  include <cuda/std/__ranges/concepts.h>
#  include <cuda/std/__ranges/subrange.h>
#endif // _LIBCUDACXX_HAS_RANGES
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_assignable.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_default_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_assignable.h>
#include <cuda/std/__type_traits/is_nothrow_convertible.h>
#include <cuda/std/__type_traits/is_nothrow_copy_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_default_constructible.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/declval.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Iter, class = void>
inline constexpr bool __noexcept_rev_iter_iter_move = false;

template <class _Iter>
inline constexpr bool __noexcept_rev_iter_iter_move<_Iter, void_t<decltype(--_CUDA_VSTD::declval<_Iter&>())>> =
  is_nothrow_copy_constructible_v<_Iter> && noexcept(_CUDA_VRANGES::iter_move(--_CUDA_VSTD::declval<_Iter&>()));

template <class _Iter, class _Iter2, class = void>
inline constexpr bool __noexcept_rev_iter_iter_swap = false;

template <class _Iter, class _Iter2>
inline constexpr bool __noexcept_rev_iter_iter_swap<_Iter, _Iter2, enable_if_t<indirectly_swappable<_Iter, _Iter2>>> =
  is_nothrow_copy_constructible_v<_Iter> && is_nothrow_copy_constructible_v<_Iter2>
  && noexcept(_CUDA_VRANGES::iter_swap(--declval<_Iter&>(), --declval<_Iter2&>()));

_LIBCUDACXX_BEGIN_HIDDEN_FRIEND_NAMESPACE

_CCCL_SUPPRESS_DEPRECATED_PUSH
template <class _Iter>
class _CCCL_TYPE_VISIBILITY_DEFAULT reverse_iterator
{
private:
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
  using pointer          = typename iterator_traits<_Iter>::pointer;
  using iterator_concept = _If<random_access_iterator<_Iter>, random_access_iterator_tag, bidirectional_iterator_tag>;
  using value_type       = iter_value_t<_Iter>;
  using difference_type  = iter_difference_t<_Iter>;
  using reference        = iter_reference_t<_Iter>;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _It2 = _Iter)
  _CCCL_REQUIRES(is_default_constructible_v<_It2>)
  _CCCL_API constexpr reverse_iterator() noexcept(is_nothrow_default_constructible_v<_It2>)
      : current()
  {}

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr explicit reverse_iterator(_Iter __x) noexcept(is_nothrow_copy_constructible_v<_Iter>)
      : current(__x)
  {}

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES((!is_same_v<_Up, _Iter>) _CCCL_AND is_convertible_v<_Up const&, _Iter>)
  _CCCL_API constexpr reverse_iterator(const reverse_iterator<_Up>& __u) noexcept(
    is_nothrow_convertible_v<_Up const&, _Iter>)
      : current(__u.base())
  {}

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES((
    !is_same_v<_Up, _Iter>) _CCCL_AND is_convertible_v<_Up const&, _Iter> _CCCL_AND is_assignable_v<_Iter&, _Up const&>)
  _CCCL_API constexpr reverse_iterator&
  operator=(const reverse_iterator<_Up>& __u) noexcept(is_nothrow_assignable_v<_Iter&, _Up const&>)
  {
    current = __u.base();
    return *this;
  }

  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr _Iter base() const noexcept(is_nothrow_copy_constructible_v<_Iter>)
  {
    return current;
  }

  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr reference operator*() const
  {
    _Iter __tmp = current;
    return *--__tmp;
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Iter2 = _Iter)
  _CCCL_REQUIRES(__has_const_arrow<_Iter2>)
  _CCCL_API constexpr pointer operator->() const
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

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr reverse_iterator& operator++()
  {
    --current;
    return *this;
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr reverse_iterator operator++(int)
  {
    reverse_iterator __tmp{*this};
    --current;
    return __tmp;
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr reverse_iterator& operator--()
  {
    ++current;
    return *this;
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr reverse_iterator operator--(int)
  {
    reverse_iterator __tmp{*this};
    ++current;
    return __tmp;
  }

  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr reverse_iterator operator+(difference_type __n) const
  {
    return reverse_iterator{current - __n};
  }

  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API friend constexpr reverse_iterator operator+(difference_type __n, const reverse_iterator& __x)
  {
    return reverse_iterator{__x.base() - __n};
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr reverse_iterator& operator+=(difference_type __n)
  {
    current -= __n;
    return *this;
  }

  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr reverse_iterator operator-(difference_type __n) const
  {
    return reverse_iterator{current + __n};
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2>
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator-(const reverse_iterator& __x, const reverse_iterator<_Iter2>& __y) -> decltype(__y.base() - __x.base())
  {
    return __y.base() - __x.base();
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr reverse_iterator& operator-=(difference_type __n)
  {
    current += __n;
    return *this;
  }

  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr reference operator[](difference_type __n) const
  {
    return *(*this + __n);
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _CCCL_API friend constexpr iter_rvalue_reference_t<_Iter2>
  iter_move(const reverse_iterator& __i) noexcept(__noexcept_rev_iter_iter_move<_Iter2>)
  {
    auto __tmp = __i.base();
    return _CUDA_VRANGES::iter_move(--__tmp);
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2>
  _CCCL_API friend constexpr auto iter_swap(const reverse_iterator& __x, const reverse_iterator<_Iter2>& __y) noexcept(
    __noexcept_rev_iter_iter_swap<_Iter, _Iter2>) _CCCL_TRAILING_REQUIRES(void)(indirectly_swappable<_Iter2, _Iter>)
  {
    auto __xtmp = __x.base();
    auto __ytmp = __y.base();
    return _CUDA_VRANGES::iter_swap(--__xtmp, --__ytmp);
  }

  template <class _Iter2>
  [[nodiscard]] _CCCL_API friend constexpr auto operator==(
    const reverse_iterator& __x, const reverse_iterator<_Iter2>& __y) noexcept(noexcept(bool(__x.base() == __y.base())))
    -> decltype(static_cast<bool>(__x.base() == __y.base()))
  {
    return __x.base() == __y.base();
  }

#if _CCCL_STD_VER <= 2017
  template <class _Iter2>
  [[nodiscard]] _CCCL_API friend constexpr auto operator!=(
    const reverse_iterator& __x, const reverse_iterator<_Iter2>& __y) noexcept(noexcept(bool(__x.base() != __y.base())))
    -> decltype(static_cast<bool>(__x.base() != __y.base()))
  {
    return __x.base() != __y.base();
  }
#endif // _CCCL_STD_VER <= 2017

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Iter2)
  _CCCL_REQUIRES(three_way_comparable_with<_Iter, _Iter2>)
  [[nodiscard]] _CCCL_API friend constexpr compare_three_way_result_t<_Iter, _Iter2>
  operator<=>(const reverse_iterator& __x, const reverse_iterator<_Iter2>& __y)
  {
    return __y.base() <=> __x.base();
  }
#else // ^^^ _LIBCUDACXX_HAS_SPACESHIP_OPERATOR() ^^^ / vvv !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR() vvv

  template <class _Iter2>
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator<(const reverse_iterator& __x, const reverse_iterator<_Iter2>& __y)
    -> decltype(static_cast<bool>(__x.base() > __y.base()))
  {
    return __x.base() > __y.base();
  }

  template <class _Iter2>
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator>(const reverse_iterator& __x, const reverse_iterator<_Iter2>& __y)
    -> decltype(static_cast<bool>(__x.base() < __y.base()))
  {
    return __x.base() < __y.base();
  }

  template <class _Iter2>
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator>=(const reverse_iterator& __x, const reverse_iterator<_Iter2>& __y)
    -> decltype(static_cast<bool>(__x.base() <= __y.base()))
  {
    return __x.base() <= __y.base();
  }

  template <class _Iter2>
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator<=(const reverse_iterator& __x, const reverse_iterator<_Iter2>& __y)
    -> decltype(static_cast<bool>(__x.base() >= __y.base()))
  {
    return __x.base() >= __y.base();
  }
#endif // !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
};
_CCCL_SUPPRESS_DEPRECATED_POP

_LIBCUDACXX_END_HIDDEN_FRIEND_NAMESPACE(reverse_iterator)

template <class _Iter1, class _Iter2>
inline constexpr bool disable_sized_sentinel_for<reverse_iterator<_Iter1>, reverse_iterator<_Iter2>> =
  !sized_sentinel_for<_Iter1, _Iter2>;

template <class _Iter>
[[nodiscard]] _CCCL_API constexpr reverse_iterator<_Iter>
make_reverse_iterator(_Iter __i) noexcept(is_nothrow_copy_constructible_v<_Iter>)
{
  return reverse_iterator<_Iter>{__i};
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ITERATOR_REVERSE_ITERATOR_H
