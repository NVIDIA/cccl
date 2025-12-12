// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___ITERATOR_MOVE_ITERATOR_H
#define _CUDA_STD___ITERATOR_MOVE_ITERATOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#  include <cuda/std/__compare/compare_three_way_result.h>
#  include <cuda/std/__compare/three_way_comparable.h>
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#include <cuda/std/__concepts/assignable.h>
#include <cuda/std/__concepts/convertible_to.h>
#include <cuda/std/__concepts/derived_from.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/incrementable_traits.h>
#include <cuda/std/__iterator/iter_move.h>
#include <cuda/std/__iterator/iter_swap.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__iterator/move_sentinel.h>
#include <cuda/std/__iterator/readable_traits.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_default_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_convertible.h>
#include <cuda/std/__type_traits/is_nothrow_copy_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_default_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__type_traits/is_reference.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__utility/ctad_support.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if _CCCL_HAS_CONCEPTS()
template <class _Iter, class = void>
struct __move_iter_category_base
{};

template <class _Iter>
  requires requires { typename iterator_traits<_Iter>::iterator_category; }
struct __move_iter_category_base<_Iter>
{
  using iterator_category =
    _If<derived_from<typename iterator_traits<_Iter>::iterator_category, random_access_iterator_tag>,
        random_access_iterator_tag,
        typename iterator_traits<_Iter>::iterator_category>;
};

template <class _Iter, class _Sent>
concept __move_iter_comparable = requires {
  { declval<const _Iter&>() == declval<_Sent>() } -> convertible_to<bool>;
};

template <class _Iter>
inline constexpr bool __noexcept_move_iter_iter_move =
  noexcept(::cuda::std::ranges::iter_move(::cuda::std::declval<const _Iter&>()));
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
template <class _Iter, class = void>
struct __move_iter_category_base
{};

template <class _Iter>
struct __move_iter_category_base<_Iter, enable_if_t<__has_iter_category<iterator_traits<_Iter>>>>
{
  using iterator_category =
    _If<derived_from<typename iterator_traits<_Iter>::iterator_category, random_access_iterator_tag>,
        random_access_iterator_tag,
        typename iterator_traits<_Iter>::iterator_category>;
};

template <class _Iter, class _Sent>
_CCCL_CONCEPT_FRAGMENT(
  __move_iter_comparable_,
  requires()(requires(convertible_to<decltype(declval<const _Iter&>() == declval<_Sent>()), bool>)));

template <class _Iter, class _Sent>
_CCCL_CONCEPT __move_iter_comparable = _CCCL_FRAGMENT(__move_iter_comparable_, _Iter, _Sent);

template <class _Iter>
inline constexpr bool __noexcept_move_iter_iter_move =
  noexcept(::cuda::std::ranges::iter_move(::cuda::std::declval<const _Iter&>()));
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

_LIBCUDACXX_BEGIN_HIDDEN_FRIEND_NAMESPACE

template <class _Iter>
class _CCCL_TYPE_VISIBILITY_DEFAULT move_iterator : public __move_iter_category_base<_Iter>
{
private:
  template <class _It2>
  friend class move_iterator;

  _Iter __current_;

  _CCCL_API static constexpr auto __mi_get_iter_concept()
  {
    if constexpr (random_access_iterator<_Iter>)
    {
      return random_access_iterator_tag{};
    }
    else if constexpr (bidirectional_iterator<_Iter>)
    {
      return bidirectional_iterator_tag{};
    }
    else if constexpr (forward_iterator<_Iter>)
    {
      return forward_iterator_tag{};
    }
    else
    {
      return input_iterator_tag{};
    }
  }

public:
  using iterator_type    = _Iter;
  using iterator_concept = decltype(__mi_get_iter_concept());

  // iterator_category is inherited and not always present
  using value_type      = iter_value_t<_Iter>;
  using difference_type = iter_difference_t<_Iter>;
  using pointer         = _Iter;
  using reference       = iter_rvalue_reference_t<_Iter>;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr explicit move_iterator(_Iter __i) noexcept(is_nothrow_move_constructible_v<_Iter>)
      : __current_(::cuda::std::move(__i))
  {}

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _It2 = _Iter)
  _CCCL_REQUIRES(is_default_constructible_v<_It2>)
  _CCCL_API constexpr move_iterator() noexcept(is_nothrow_default_constructible_v<_It2>)
      : __current_()
  {}

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES((!is_same_v<_Up, _Iter>) _CCCL_AND convertible_to<const _Up&, _Iter>)
  _CCCL_API constexpr move_iterator(const move_iterator<_Up>& __u) noexcept(is_nothrow_convertible_v<const _Up&, _Iter>)
      : __current_(__u.base())
  {}

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES(
    (!is_same_v<_Up, _Iter>) _CCCL_AND convertible_to<const _Up&, _Iter> _CCCL_AND assignable_from<_Iter&, const _Up&>)
  _CCCL_API constexpr move_iterator&
  operator=(const move_iterator<_Up>& __u) noexcept(is_nothrow_assignable_v<_Iter&, const _Up&>)
  {
    __current_ = __u.base();
    return *this;
  }

  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr const _Iter& base() const& noexcept
  {
    return __current_;
  }

  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr _Iter base() &&
  {
    return ::cuda::std::move(__current_);
  }

  [[nodiscard]] _CCCL_API constexpr reference operator*() const
  {
    return ::cuda::std::ranges::iter_move(__current_);
  }

  [[nodiscard]] _CCCL_API constexpr reference operator[](difference_type __n) const
  {
    return ::cuda::std::ranges::iter_move(__current_ + __n);
  }

  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_DEPRECATED_IN_CXX20 _CCCL_API constexpr pointer operator->() const
  {
    return __current_;
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr move_iterator& operator++()
  {
    ++__current_;
    return *this;
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr auto operator++(int)
  {
    if constexpr (forward_iterator<_Iter>)
    {
      move_iterator __tmp{*this};
      ++__current_;
      return __tmp;
    }
    else
    {
      ++__current_;
    }
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr move_iterator& operator--()
  {
    --__current_;
    return *this;
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr move_iterator operator--(int)
  {
    move_iterator __tmp{*this};
    --__current_;
    return __tmp;
  }

  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr move_iterator operator+(difference_type __n) const
  {
    return move_iterator{__current_ + __n};
  }

  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API friend constexpr move_iterator operator+(difference_type __n, const move_iterator& __x)
  {
    return move_iterator{__x.base() + __n};
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr move_iterator& operator+=(difference_type __n)
  {
    __current_ += __n;
    return *this;
  }

  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr move_iterator operator-(difference_type __n) const
  {
    return move_iterator{__current_ - __n};
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2>
  [[nodiscard]] _CCCL_API friend constexpr auto operator-(const move_iterator& __x, const move_iterator<_Iter2>& __y)
    -> decltype(::cuda::std::declval<const _Iter&>() - ::cuda::std::declval<const _Iter2&>())
  {
    return __x.base() - __y.base();
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr move_iterator& operator-=(difference_type __n)
  {
    __current_ -= __n;
    return *this;
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Sent)
  _CCCL_REQUIRES(sentinel_for<_Sent, _Iter> _CCCL_AND __move_iter_comparable<_Iter, _Sent>)
  [[nodiscard]] _CCCL_API friend constexpr bool operator==(const move_iterator& __x, const move_sentinel<_Sent>& __y)
  {
    return __x.base() == __y.base();
  }

#if _CCCL_STD_VER < 2020
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Sent)
  _CCCL_REQUIRES(sentinel_for<_Sent, _Iter> _CCCL_AND __move_iter_comparable<_Iter, _Sent>)
  [[nodiscard]] _CCCL_API friend constexpr bool operator==(const move_sentinel<_Sent>& __y, const move_iterator& __x)
  {
    return __y.base() == __x.base();
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Sent)
  _CCCL_REQUIRES(sentinel_for<_Sent, _Iter> _CCCL_AND __move_iter_comparable<_Iter, _Sent>)
  [[nodiscard]] _CCCL_API friend constexpr bool operator!=(const move_iterator& __x, const move_sentinel<_Sent>& __y)
  {
    return __x.base() != __y.base();
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Sent)
  _CCCL_REQUIRES(sentinel_for<_Sent, _Iter> _CCCL_AND __move_iter_comparable<_Iter, _Sent>)
  [[nodiscard]] _CCCL_API friend constexpr bool operator!=(const move_sentinel<_Sent>& __y, const move_iterator& __x)
  {
    return __y.base() != __x.base();
  }
#endif // _CCCL_STD_VER < 2020

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2>
  [[nodiscard]] _CCCL_API friend constexpr auto operator==(const move_iterator& __x, const move_iterator<_Iter2>& __y)
    -> decltype(static_cast<bool>(::cuda::std::declval<const _Iter&>() == ::cuda::std::declval<const _Iter2&>()))
  {
    return __x.base() == __y.base();
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2>
  [[nodiscard]] _CCCL_API friend constexpr auto operator!=(const move_iterator& __x, const move_iterator<_Iter2>& __y)
    -> decltype(static_cast<bool>(::cuda::std::declval<const _Iter&>() != ::cuda::std::declval<const _Iter2&>()))
  {
    return __x.base() != __y.base();
  }

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Iter2)
  _CCCL_REQUIRES(three_way_comparable_with<_Iter, _Iter2>)
  [[nodiscard]] _CCCL_API friend constexpr compare_three_way_result_t<_Iter, _Iter2>
  operator<=>(const move_iterator& __x, const move_iterator<_Iter2>& __y)
  {
    return __x.base() <=> __y.base();
  }

#else // ^^^ _LIBCUDACXX_HAS_SPACESHIP_OPERATOR() ^^^ / vvv !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR() vvv
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2>
  [[nodiscard]] _CCCL_API friend constexpr auto operator<(const move_iterator& __x, const move_iterator<_Iter2>& __y)
    -> decltype(static_cast<bool>(::cuda::std::declval<const _Iter&>() < ::cuda::std::declval<const _Iter2&>()))
  {
    return __x.base() < __y.base();
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2>
  [[nodiscard]] _CCCL_API friend constexpr auto operator>(const move_iterator& __x, const move_iterator<_Iter2>& __y)
    -> decltype(static_cast<bool>(::cuda::std::declval<const _Iter&>() > ::cuda::std::declval<const _Iter2&>()))
  {
    return __x.base() > __y.base();
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2>
  [[nodiscard]] _CCCL_API friend constexpr auto operator<=(const move_iterator& __x, const move_iterator<_Iter2>& __y)
    -> decltype(static_cast<bool>(::cuda::std::declval<const _Iter&>() <= ::cuda::std::declval<const _Iter2&>()))
  {
    return __x.base() <= __y.base();
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2>
  [[nodiscard]] _CCCL_API friend constexpr auto operator>=(const move_iterator& __x, const move_iterator<_Iter2>& __y)
    -> decltype(static_cast<bool>(::cuda::std::declval<const _Iter&>() >= ::cuda::std::declval<const _Iter2&>()))
  {
    return __x.base() >= __y.base();
  }
#endif // !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Sent)
  _CCCL_REQUIRES(sized_sentinel_for<_Sent, _Iter>)
  [[nodiscard]] _CCCL_API friend constexpr iter_difference_t<_Iter>
  operator-(const move_sentinel<_Sent>& __x, const move_iterator& __y)
  {
    return __x.base() - __y.base();
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Sent)
  _CCCL_REQUIRES(sized_sentinel_for<_Sent, _Iter>)
  [[nodiscard]] _CCCL_API friend constexpr iter_difference_t<_Iter>
  operator-(const move_iterator& __x, const move_sentinel<_Sent>& __y)
  {
    return __x.base() - __y.base();
  }

  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API friend constexpr iter_rvalue_reference_t<_Iter>
  iter_move(const move_iterator& __i) noexcept(__noexcept_move_iter_iter_move<_Iter>)
  {
    return ::cuda::std::ranges::iter_move(__i.__current_);
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2>
  _CCCL_API friend constexpr auto
  iter_swap(const move_iterator& __x, const move_iterator<_Iter2>& __y) noexcept(__noexcept_swappable<_Iter, _Iter2>)
    _CCCL_TRAILING_REQUIRES(void)(indirectly_swappable<_Iter2, _Iter>)
  {
    return ::cuda::std::ranges::iter_swap(__x.__current_, __y.__current_);
  }
};
_CCCL_CTAD_SUPPORTED_FOR_TYPE(move_iterator);
_LIBCUDACXX_END_HIDDEN_FRIEND_NAMESPACE(move_iterator)

// Some compilers have issues determining __is_fancy_pointer
#if _CCCL_COMPILER(GCC) || _CCCL_COMPILER(MSVC)
template <class _Iter>
inline constexpr bool __is_fancy_pointer<move_iterator<_Iter>> = __is_fancy_pointer<_Iter>;
#endif // _CCCL_COMPILER(GCC) || _CCCL_COMPILER(MSVC)

_CCCL_EXEC_CHECK_DISABLE
template <class _Iter>
[[nodiscard]] _CCCL_API constexpr move_iterator<_Iter>
make_move_iterator(_Iter __i) noexcept(is_nothrow_move_constructible_v<_Iter>)
{
  return move_iterator<_Iter>{::cuda::std::move(__i)};
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ITERATOR_MOVE_ITERATOR_H
