// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCUDACXX___RANGES_IOTA_VIEW_H
#define _LIBCUDACXX___RANGES_IOTA_VIEW_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__assert"
#ifndef _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR
#include "../__compare/three_way_comparable.h"
#endif
#include "../__concepts/arithmetic.h"
#include "../__concepts/constructible.h"
#include "../__concepts/convertible_to.h"
#include "../__concepts/copyable.h"
#include "../__concepts/equality_comparable.h"
#include "../__concepts/invocable.h"
#include "../__concepts/same_as.h"
#include "../__concepts/semiregular.h"
#include "../__concepts/totally_ordered.h"
#include "../__functional/ranges_operations.h"
#include "../__iterator/concepts.h"
#include "../__iterator/incrementable_traits.h"
#include "../__iterator/iterator_traits.h"
#include "../__iterator/unreachable_sentinel.h"
#include "../__ranges/copyable_box.h"
#include "../__ranges/enable_borrowed_range.h"
#include "../__ranges/view_interface.h"
#include "../__type_traits/conditional.h"
#include "../__type_traits/enable_if.h"
#include "../__type_traits/is_constructible.h"
#include "../__type_traits/is_nothrow_copy_constructible.h"
#include "../__type_traits/make_unsigned.h"
#include "../__type_traits/type_identity.h"
#include "../__type_traits/void_t.h"
#include "../__utility/forward.h"
#include "../__utility/move.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif


#if _LIBCUDACXX_STD_VER > 14

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES
_LIBCUDACXX_BEGIN_NAMESPACE_RANGES_ABI

  template<class _Int>
  struct __get_wider_signed {
    _LIBCUDACXX_INLINE_VISIBILITY static auto __call() {
           if constexpr (sizeof(_Int) < sizeof(short)) return type_identity<short>{};
      else if constexpr (sizeof(_Int) < sizeof(int))   return type_identity<int>{};
      else if constexpr (sizeof(_Int) < sizeof(long))  return type_identity<long>{};
      else                                             return type_identity<long long>{};

      static_assert(sizeof(_Int) <= sizeof(long long),
        "Found integer-like type that is bigger than largest integer like type.");
      _LIBCUDACXX_UNREACHABLE();
    }

    using type = typename decltype(__call())::type;
  };

  template<class _Start>
  using _IotaDiffT = typename _If<
      (!integral<_Start> || sizeof(iter_difference_t<_Start>) > sizeof(_Start)),
      type_identity<iter_difference_t<_Start>>,
      __get_wider_signed<_Start>
    >::type;

#if _LIBCUDACXX_STD_VER > 17
  template<class _Iter>
  concept __decrementable = incrementable<_Iter> && requires(_Iter __i) {
    { --__i } -> same_as<_Iter&>;
    { __i-- } -> same_as<_Iter>;
  };

  template<class _Iter>
  concept __advanceable =
    __decrementable<_Iter> && totally_ordered<_Iter> &&
    requires(_Iter __i, const _Iter __j, const _IotaDiffT<_Iter> __n) {
      { __i += __n } -> same_as<_Iter&>;
      { __i -= __n } -> same_as<_Iter&>;
      _Iter(__j + __n);
      _Iter(__n + __j);
      _Iter(__j - __n);
      { __j - __j } -> convertible_to<_IotaDiffT<_Iter>>;
    };
#else
  template <class _Iter>
  _LIBCUDACXX_CONCEPT_FRAGMENT(
    __decrementable_,
    requires(_Iter __i)(
      requires(incrementable<_Iter>),
      requires(same_as<decltype(--__i), _Iter&>),
      requires(same_as<decltype(__i--), _Iter>)
    ));

  template <class _Iter>
  _LIBCUDACXX_CONCEPT __decrementable = _LIBCUDACXX_FRAGMENT(__decrementable_, _Iter);

  template <class _Iter>
  _LIBCUDACXX_CONCEPT_FRAGMENT(
    __advanceable_,
    requires(_Iter __i, const _Iter __j, const _IotaDiffT<_Iter> __n)(
      requires(__decrementable<_Iter>),
      requires(totally_ordered<_Iter>),
      requires(same_as<decltype(__i += __n), _Iter&>),
      requires(same_as<decltype(__i -= __n), _Iter&>),
      requires(is_constructible_v<_Iter, decltype(__j + __n)>),
      requires(is_constructible_v<_Iter, decltype(__n + __j)>),
      requires(is_constructible_v<_Iter, decltype(__j - __n)>),
      requires(convertible_to<decltype(__j - __j),_IotaDiffT<_Iter>>)
    ));

  template <class _Iter>
  _LIBCUDACXX_CONCEPT __advanceable = _LIBCUDACXX_FRAGMENT(__advanceable_, _Iter);
#endif // _LIBCUDACXX_STD_VER < 20

  template<class, class = void>
  struct __iota_iterator_category {};

  template<class _Tp>
  struct __iota_iterator_category<_Tp, enable_if_t<incrementable<_Tp>>> {
    using iterator_category = input_iterator_tag;
  };

#if _LIBCUDACXX_STD_VER > 17
  template <weakly_incrementable _Start, semiregular _BoundSentinel = unreachable_sentinel_t>
    requires __weakly_equality_comparable_with<_Start, _BoundSentinel> && copyable<_Start>
#else
  template <class _Start, class _BoundSentinel = unreachable_sentinel_t,
    enable_if_t<weakly_incrementable<_Start>, int> = 0,
    enable_if_t<semiregular<_BoundSentinel>, int> = 0,
    enable_if_t<__weakly_equality_comparable_with<_Start, _BoundSentinel>, int> = 0,
    enable_if_t<copyable<_Start>, int> = 0>
#endif
  class iota_view : public view_interface<iota_view<_Start, _BoundSentinel>> {
  public:
    struct __iterator : public __iota_iterator_category<_Start> {
      friend class iota_view;

      using iterator_concept =
        _If<__advanceable<_Start>,   random_access_iterator_tag,
        _If<__decrementable<_Start>, bidirectional_iterator_tag,
        _If<incrementable<_Start>,   forward_iterator_tag,
        /*Else*/                     input_iterator_tag>>>;

      using value_type = _Start;
      using difference_type = _IotaDiffT<_Start>;

      _Start __value_ = _Start();

#if _LIBCUDACXX_STD_VER > 17
      _LIBCUDACXX_HIDE_FROM_ABI
      __iterator() requires default_initializable<_Start> = default;
#else
      _LIBCUDACXX_TEMPLATE(class _Start2 = _Start)
        (requires default_initializable<_Start2>)
      _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
      constexpr __iterator() noexcept(is_nothrow_default_constructible_v<_Start2>) {}
#endif

      _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
      constexpr explicit __iterator(_Start __value) : __value_(_CUDA_VSTD::move(__value)) {}

      _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
      constexpr _Start operator*() const noexcept(is_nothrow_copy_constructible_v<_Start>) {
        return __value_;
      }

      _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
      constexpr __iterator& operator++() {
        ++__value_;
        return *this;
      }

      _LIBCUDACXX_TEMPLATE(class _Start2 = _Start)
        (requires (!incrementable<_Start2>))
      _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
      constexpr void operator++(int) { ++*this; }

      _LIBCUDACXX_TEMPLATE(class _Start2 = _Start)
        (requires incrementable<_Start2>)
      _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
      constexpr __iterator operator++(int) {
        auto __tmp = *this;
        ++*this;
        return __tmp;
      }

      _LIBCUDACXX_TEMPLATE(class _Start2 = _Start)
        (requires __decrementable<_Start2>)
      _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
      constexpr __iterator& operator--() {
        --__value_;
        return *this;
      }

      _LIBCUDACXX_TEMPLATE(class _Start2 = _Start)
        (requires __decrementable<_Start2>)
      _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
      constexpr __iterator  operator--(int) {
        auto __tmp = *this;
        --*this;
        return __tmp;
      }

      _LIBCUDACXX_TEMPLATE(class _Start2 = _Start)
        (requires __advanceable<_Start2>)
      _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
      constexpr __iterator& operator+=(difference_type __n)
      {
        if constexpr (__integer_like<_Start> && !__signed_integer_like<_Start>) {
          if (__n >= difference_type(0)) {
            __value_ += static_cast<_Start>(__n);
          } else {
            __value_ -= static_cast<_Start>(-__n);
          }
        } else if constexpr (__signed_integer_like<_Start>) {
          __value_ += static_cast<_Start>(__n);;
        } else {
          __value_ += __n;
        }
        return *this;
      }

      _LIBCUDACXX_TEMPLATE(class _Start2 = _Start)
        (requires __advanceable<_Start2>)
      _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
      constexpr __iterator& operator-=(difference_type __n)
      {
        if constexpr (__integer_like<_Start> && !__signed_integer_like<_Start>) {
          if (__n >= difference_type(0)) {
            __value_ -= static_cast<_Start>(__n);
          } else {
            __value_ += static_cast<_Start>(-__n);
          }
        } else if constexpr (__signed_integer_like<_Start>) {
          __value_ -= static_cast<_Start>(__n);;
        } else {
          __value_ -= __n;
        }
        return *this;
      }

      _LIBCUDACXX_TEMPLATE(class _Start2 = _Start)
        (requires __advanceable<_Start2>)
      _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
      constexpr _Start2 operator[](difference_type __n) const
      {
        return _Start(__value_ + __n);
      }

      _LIBCUDACXX_TEMPLATE(class _Start2 = _Start)
        (requires equality_comparable<_Start2>)
      _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
      friend constexpr bool operator==(const __iterator& __x, const __iterator& __y)
      {
        return __x.__value_ == __y.__value_;
      }

#if _LIBCUDACXX_STD_VER < 20
      _LIBCUDACXX_TEMPLATE(class _Start2 = _Start)
        (requires equality_comparable<_Start2>)
      _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
      friend constexpr bool operator!=(const __iterator& __x, const __iterator& __y)
      {
        return __x.__value_ != __y.__value_;
      }
#endif

      _LIBCUDACXX_TEMPLATE(class _Start2 = _Start)
        (requires totally_ordered<_Start2>)
      _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
      friend constexpr bool operator<(const __iterator& __x, const __iterator& __y)
      {
        return __x.__value_ < __y.__value_;
      }

      _LIBCUDACXX_TEMPLATE(class _Start2 = _Start)
        (requires totally_ordered<_Start2>)
      _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
      friend constexpr bool operator>(const __iterator& __x, const __iterator& __y)
      {
        return __y < __x;
      }

      _LIBCUDACXX_TEMPLATE(class _Start2 = _Start)
        (requires totally_ordered<_Start2>)
      _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
      friend constexpr bool operator<=(const __iterator& __x, const __iterator& __y)
      {
        return !(__y < __x);
      }

      _LIBCUDACXX_TEMPLATE(class _Start2 = _Start)
        (requires totally_ordered<_Start2>)
      _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
      friend constexpr bool operator>=(const __iterator& __x, const __iterator& __y)
      {
        return !(__x < __y);
      }

#ifndef _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR
      _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
      friend constexpr auto operator<=>(const __iterator& __x, const __iterator& __y)
        requires totally_ordered<_Start> && three_way_comparable<_Start>
      {
        return __x.__value_ <=> __y.__value_;
      }
#endif // _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR

      _LIBCUDACXX_TEMPLATE(class _Start2 = _Start)
        (requires __advanceable<_Start2>)
      _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
      friend constexpr __iterator operator+(__iterator __i, difference_type __n)
      {
        __i += __n;
        return __i;
      }

      _LIBCUDACXX_TEMPLATE(class _Start2 = _Start)
        (requires __advanceable<_Start2>)
      _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
      friend constexpr __iterator operator+(difference_type __n, __iterator __i)
      {
        return __i + __n;
      }

      _LIBCUDACXX_TEMPLATE(class _Start2 = _Start)
        (requires __advanceable<_Start2>)
      _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
      friend constexpr __iterator operator-(__iterator __i, difference_type __n)
      {
        __i -= __n;
        return __i;
      }

      _LIBCUDACXX_TEMPLATE(class _Start2 = _Start)
        (requires __advanceable<_Start2>)
      _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
      friend constexpr difference_type operator-(const __iterator& __x, const __iterator& __y)
      {
        if constexpr (__integer_like<_Start> && !__signed_integer_like<_Start>) {
          if (__y.__value_ > __x.__value_) {
            return static_cast<difference_type>(-static_cast<difference_type>(__y.__value_ - __x.__value_));
          }
          return static_cast<difference_type>(__x.__value_ - __y.__value_);
        } else if constexpr(__signed_integer_like<_Start>) {
          return static_cast<difference_type>(static_cast<difference_type>(__x.__value_) - static_cast<difference_type>(__y.__value_));
        }
#if !defined(_LIBCUDACXX_COMPILER_NVHPC) // nvhpc cannot compile with this else
        else
#endif // !_LIBCUDACXX_COMPILER_NVHPC
        {
          return __x.__value_ - __y.__value_;
        }
        _LIBCUDACXX_UNREACHABLE();
      }
    };

    struct __sentinel {
      friend class iota_view;

    private:
      _BoundSentinel __bound_sentinel_ = _BoundSentinel();

    public:
      _LIBCUDACXX_HIDE_FROM_ABI
      __sentinel() = default;
      _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
      constexpr explicit __sentinel(_BoundSentinel __bound_sentinel) : __bound_sentinel_(_CUDA_VSTD::move(__bound_sentinel)) {}

      _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
      friend constexpr bool operator==(const __iterator& __x, const __sentinel& __y) {
        return __x.__value_ == __y.__bound_sentinel_;
      }
#if _LIBCUDACXX_STD_VER < 20
      _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
      friend constexpr bool operator==(const __sentinel& __x, const __iterator& __y) {
        return __x.__bound_sentinel_ == __y.__value_;
      }

      _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
      friend constexpr bool operator!=(const __iterator& __x, const __sentinel& __y) {
        return __x.__value_ != __y.__bound_sentinel_;
      }

      _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
      friend constexpr bool operator!=(const __sentinel& __x, const __iterator& __y) {
        return __x.__bound_sentinel_ != __y.__value_;
      }
#endif

      _LIBCUDACXX_TEMPLATE(class _BoundSentinel2 = _BoundSentinel)
        (requires sized_sentinel_for<_BoundSentinel2, _Start>)
      _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
      friend constexpr iter_difference_t<_Start> operator-(const __iterator& __x, const __sentinel& __y)
      {
        return __x.__value_ - __y.__bound_sentinel_;
      }

      _LIBCUDACXX_TEMPLATE(class _BoundSentinel2 = _BoundSentinel)
        (requires sized_sentinel_for<_BoundSentinel2, _Start>)
      _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
      friend constexpr iter_difference_t<_Start> operator-(const __sentinel& __x, const __iterator& __y)
      {
        return -(__y - __x);
      }
    };
  private:
    _Start __value_ = _Start();
    _BoundSentinel __bound_sentinel_ = _BoundSentinel();

  public:
#if _LIBCUDACXX_STD_VER > 17
    _LIBCUDACXX_HIDE_FROM_ABI
    iota_view() requires default_initializable<_Start> = default;
#else
    _LIBCUDACXX_TEMPLATE(class _Start2 = _Start)
      (requires default_initializable<_Start2>)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr iota_view() noexcept(is_nothrow_default_constructible_v<_Start2>)
      : view_interface<iota_view<_Start, _BoundSentinel>>() {}
#endif

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr explicit iota_view(_Start __value)
      : view_interface<iota_view<_Start, _BoundSentinel>>()
      , __value_(_CUDA_VSTD::move(__value)) { }

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr iota_view(type_identity_t<_Start> __value, type_identity_t<_BoundSentinel> __bound_sentinel)
      : view_interface<iota_view<_Start, _BoundSentinel>>()
      , __value_(_CUDA_VSTD::move(__value))
      , __bound_sentinel_(_CUDA_VSTD::move(__bound_sentinel)) {
      // Validate the precondition if possible.
      if constexpr (totally_ordered_with<_Start, _BoundSentinel>) {
        _LIBCUDACXX_ASSERT(_CUDA_VRANGES::less_equal()(__value_, __bound_sentinel_),
                       "Precondition violated: value is greater than bound.");
      }
    }

    _LIBCUDACXX_TEMPLATE(class _BoundSentinel2 = _BoundSentinel)
      (requires same_as<_Start, _BoundSentinel2>)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr iota_view(__iterator __first, __iterator __last)
      : iota_view(_CUDA_VSTD::move(__first.__value_), _CUDA_VSTD::move(__last.__value_)) {}

    _LIBCUDACXX_TEMPLATE(class _BoundSentinel2 = _BoundSentinel)
      (requires same_as<_BoundSentinel2, unreachable_sentinel_t>)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr iota_view(__iterator __first, _BoundSentinel __last)
      : iota_view(_CUDA_VSTD::move(__first.__value_), _CUDA_VSTD::move(__last)) {}

    _LIBCUDACXX_TEMPLATE(class _BoundSentinel2 = _BoundSentinel)
      (requires (!same_as<_Start, _BoundSentinel2> && !same_as<_Start, unreachable_sentinel_t>))
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr iota_view(__iterator __first, __sentinel __last)
      : iota_view(_CUDA_VSTD::move(__first.__value_), _CUDA_VSTD::move(__last.__bound_sentinel_)) {}

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr __iterator begin() const { return __iterator{__value_}; }

    _LIBCUDACXX_TEMPLATE(class _BoundSentinel2 = _BoundSentinel)
      (requires (!same_as<_Start, _BoundSentinel2>))
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr auto end() const {
      if constexpr (same_as<_BoundSentinel, unreachable_sentinel_t>) {
        return unreachable_sentinel;
      } else {
        return __sentinel{__bound_sentinel_};
      }
      _LIBCUDACXX_UNREACHABLE();
    }

    _LIBCUDACXX_TEMPLATE(class _BoundSentinel2 = _BoundSentinel)
      (requires same_as<_Start, _BoundSentinel2>)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr __iterator end() const
    {
      return __iterator{__bound_sentinel_};
    }

    _LIBCUDACXX_TEMPLATE(class _BoundSentinel2 = _BoundSentinel)
      (requires (same_as<_Start, _BoundSentinel2> && __advanceable<_Start>) ||
                (integral<_Start> && integral<_BoundSentinel2>) || sized_sentinel_for<_BoundSentinel2, _Start>)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr auto size() const
    {
      if constexpr (__integer_like<_Start> && __integer_like<_BoundSentinel>) {
        if (__value_ < 0) {
          if (__bound_sentinel_ < 0) {
            return _CUDA_VSTD::__to_unsigned_like(-__value_) - _CUDA_VSTD::__to_unsigned_like(-__bound_sentinel_);
          }
          return _CUDA_VSTD::__to_unsigned_like(__bound_sentinel_) + _CUDA_VSTD::__to_unsigned_like(-__value_);
        }
        return _CUDA_VSTD::__to_unsigned_like(__bound_sentinel_) - _CUDA_VSTD::__to_unsigned_like(__value_);
      } else {
        return _CUDA_VSTD::__to_unsigned_like(__bound_sentinel_ - __value_);
      }
      _LIBCUDACXX_UNREACHABLE();
    }
  };

  _LIBCUDACXX_TEMPLATE(class _Start, class _BoundSentinel)
    (requires(!__integer_like<_Start> || !__integer_like<_BoundSentinel> ||
             (__signed_integer_like<_Start> == __signed_integer_like<_BoundSentinel>)))
  iota_view(_Start, _BoundSentinel) -> iota_view<_Start, _BoundSentinel>;

_LIBCUDACXX_END_NAMESPACE_RANGES_ABI

  template <class _Start, class _BoundSentinel>
  inline constexpr bool enable_borrowed_range<iota_view<_Start, _BoundSentinel>> = true;

_LIBCUDACXX_END_NAMESPACE_RANGES

_LIBCUDACXX_BEGIN_NAMESPACE_VIEWS
_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__iota)

  struct __fn {
    template<class _Start>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr auto operator()(_Start&& __start) const
      noexcept(noexcept(_CUDA_VRANGES::iota_view(_CUDA_VSTD::forward<_Start>(__start))))
      -> decltype(      _CUDA_VRANGES::iota_view(_CUDA_VSTD::forward<_Start>(__start)))
      { return          _CUDA_VRANGES::iota_view(_CUDA_VSTD::forward<_Start>(__start)); }

    template <class _Start, class _BoundSentinel>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr auto operator()(_Start&& __start, _BoundSentinel&& __bound_sentinel) const
      noexcept(noexcept(_CUDA_VRANGES::iota_view(_CUDA_VSTD::forward<_Start>(__start), _CUDA_VSTD::forward<_BoundSentinel>(__bound_sentinel))))
      -> decltype(      _CUDA_VRANGES::iota_view(_CUDA_VSTD::forward<_Start>(__start), _CUDA_VSTD::forward<_BoundSentinel>(__bound_sentinel)))
      { return          _CUDA_VRANGES::iota_view(_CUDA_VSTD::forward<_Start>(__start), _CUDA_VSTD::forward<_BoundSentinel>(__bound_sentinel)); }
  };
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo {
  _LIBCUDACXX_CPO_ACCESSIBILITY auto iota = __iota::__fn{};
} // namespace __cpo

_LIBCUDACXX_END_NAMESPACE_VIEWS

#endif // _LIBCUDACXX_STD_VER > 14


#endif // _LIBCUDACXX___RANGES_IOTA_VIEW_H
