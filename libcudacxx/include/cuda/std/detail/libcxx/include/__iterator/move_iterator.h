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

#ifndef _LIBCUDACXX___ITERATOR_MOVE_ITERATOR_H
#define _LIBCUDACXX___ITERATOR_MOVE_ITERATOR_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#ifndef _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR
#include "../__compare/compare_three_way_result.h"
#include "../__compare/three_way_comparable.h"
#endif // _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR
#include "../__concepts/assignable.h"
#include "../__concepts/convertible_to.h"
#include "../__concepts/derived_from.h"
#include "../__concepts/same_as.h"
#include "../__iterator/concepts.h"
#include "../__iterator/incrementable_traits.h"
#include "../__iterator/iter_move.h"
#include "../__iterator/iter_swap.h"
#include "../__iterator/iterator_traits.h"
#include "../__iterator/move_sentinel.h"
#include "../__iterator/readable_traits.h"
#include "../__type_traits/conditional.h"
#include "../__type_traits/is_reference.h"
#include "../__type_traits/remove_reference.h"
#include "../__utility/move.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER > 17
template<class _Iter, class = void>
struct __move_iter_category_base {};

template<class _Iter>
  requires requires { typename iterator_traits<_Iter>::iterator_category; }
struct __move_iter_category_base<_Iter> {
    using iterator_category = _If<
        derived_from<typename iterator_traits<_Iter>::iterator_category, random_access_iterator_tag>,
        random_access_iterator_tag,
        typename iterator_traits<_Iter>::iterator_category
    >;
};

template<class _Iter, class _Sent>
concept __move_iter_comparable = requires {
    { declval<const _Iter&>() == declval<_Sent>() } -> convertible_to<bool>;
};
#elif _LIBCUDACXX_STD_VER > 14
template<class _Iter, class = void>
struct __move_iter_category_base {};

template<class _Iter>
struct __move_iter_category_base<_Iter, enable_if_t<__has_iter_category<iterator_traits<_Iter>>>> {
    using iterator_category = _If<
        derived_from<typename iterator_traits<_Iter>::iterator_category, random_access_iterator_tag>,
        random_access_iterator_tag,
        typename iterator_traits<_Iter>::iterator_category
    >;
};

template<class _Iter, class _Sent>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __move_iter_comparable_,
  requires()(
    requires(convertible_to<decltype(declval<const _Iter&>() == declval<_Sent>()), bool>)
  ));

template<class _Iter, class _Sent>
_LIBCUDACXX_CONCEPT __move_iter_comparable = _LIBCUDACXX_FRAGMENT(__move_iter_comparable_, _Iter, _Sent);

#endif // _LIBCUDACXX_STD_VER > 17

template <class _Iter>
class _LIBCUDACXX_TEMPLATE_VIS move_iterator
#if _LIBCUDACXX_STD_VER > 14
    : public __move_iter_category_base<_Iter>
#endif
{
private:
    template<class _It2> friend class move_iterator;

    _Iter __current_;
public:
#if _LIBCUDACXX_STD_VER > 14
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    static constexpr auto __get_iter_concept() {
        if constexpr (random_access_iterator<_Iter>) {
            return random_access_iterator_tag{};
        } else if constexpr (bidirectional_iterator<_Iter>) {
            return bidirectional_iterator_tag{};
        } else if constexpr (forward_iterator<_Iter>) {
            return forward_iterator_tag{};
        } else {
            return input_iterator_tag{};
        }
        _LIBCUDACXX_UNREACHABLE();
    }

    using iterator_type = _Iter;
    using iterator_concept = decltype(__get_iter_concept());

    // iterator_category is inherited and not always present
    using value_type = iter_value_t<_Iter>;
    using difference_type = iter_difference_t<_Iter>;
    using pointer = _Iter;
    using reference = iter_rvalue_reference_t<_Iter>;
#else // ^^^ _LIBCUDACXX_STD_VER > 14 ^^^ / vvv _LIBCUDACXX_STD_VER < 17 vvv
    typedef _Iter iterator_type;
    typedef _If<
        __is_cpp17_random_access_iterator<_Iter>::value,
        random_access_iterator_tag,
        typename iterator_traits<_Iter>::iterator_category
    > iterator_category;
    typedef typename iterator_traits<iterator_type>::value_type value_type;
    typedef typename iterator_traits<iterator_type>::difference_type difference_type;
    typedef iterator_type pointer;

    typedef typename iterator_traits<iterator_type>::reference __reference;
    typedef __conditional_t<
            is_reference<__reference>::value,
            __libcpp_remove_reference_t<__reference>&&,
            __reference
        > reference;
#endif // _LIBCUDACXX_STD_VER < 17

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
    explicit move_iterator(_Iter __i) : __current_(_CUDA_VSTD::move(__i)) {}

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
    move_iterator& operator++() { ++__current_; return *this; }

    _LIBCUDACXX_DEPRECATED_IN_CXX20 _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 pointer operator->() const { return __current_; }

#if _LIBCUDACXX_STD_VER > 14
#if _LIBCUDACXX_STD_VER > 17
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
    move_iterator() requires is_constructible_v<_Iter> : __current_() {}
#else // ^^^ _LIBCUDACXX_STD_VER > 17 ^^^ / vvv _LIBCUDACXX_STD_VER < 20 vvv
    _LIBCUDACXX_TEMPLATE(class _It2 = _Iter)
        (requires is_constructible_v<_It2>)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
    move_iterator() : __current_() {}
#endif // _LIBCUDACXX_STD_VER < 20

    _LIBCUDACXX_TEMPLATE(class _Up)
        (requires (!_IsSame<_Up, _Iter>::value) && convertible_to<const _Up&, _Iter>)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
    move_iterator(const move_iterator<_Up>& __u) : __current_(__u.base()) {}

    _LIBCUDACXX_TEMPLATE(class _Up)
        (requires (!_IsSame<_Up, _Iter>::value) &&
                 convertible_to<const _Up&, _Iter> &&
                 assignable_from<_Iter&, const _Up&>)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
    move_iterator& operator=(const move_iterator<_Up>& __u) {
        __current_ = __u.base();
        return *this;
    }

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
    const _Iter& base() const & noexcept { return __current_; }
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
    _Iter base() && { return _CUDA_VSTD::move(__current_); }

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
    reference operator*() const { return _CUDA_VRANGES::iter_move(__current_); }
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
    reference operator[](difference_type __n) const { return _CUDA_VRANGES::iter_move(__current_ + __n); }

    _LIBCUDACXX_TEMPLATE(class _It2 = _Iter)
        (requires forward_iterator<_It2>)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
    auto operator++(int)
    {
        move_iterator __tmp(*this); ++__current_; return __tmp;
    }

    _LIBCUDACXX_TEMPLATE(class _It2 = _Iter)
        (requires (!forward_iterator<_It2>))
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
    void operator++(int) { ++__current_; }
#else // ^^^ _LIBCUDACXX_STD_VER > 14 ^^^ / vvv _LIBCUDACXX_STD_VER < 17 vvv
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
    move_iterator() : __current_() {}

    template <class _Up, class = __enable_if_t<
        !is_same<_Up, _Iter>::value && is_convertible<const _Up&, _Iter>::value
    > >
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
    move_iterator(const move_iterator<_Up>& __u) : __current_(__u.base()) {}

    template <class _Up, class = __enable_if_t<
        !is_same<_Up, _Iter>::value &&
        is_convertible<const _Up&, _Iter>::value &&
        is_assignable<_Iter&, const _Up&>::value
    > >
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
    move_iterator& operator=(const move_iterator<_Up>& __u) {
        __current_ = __u.base();
        return *this;
    }

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
    _Iter base() const { return __current_; }

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
    reference operator*() const { return static_cast<reference>(*__current_); }

#if defined(_LIBCUDACXX_COMPILER_MSVC)
#pragma warning(push)
#pragma warning(disable: 4172) // returning address of local variable or temporary
#endif // _LIBCUDACXX_COMPILER_MSVC

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
    reference operator[](difference_type __n) const { return static_cast<reference>(__current_[__n]); }

#if defined(_LIBCUDACXX_COMPILER_MSVC)
#pragma warning(pop)
#endif // _LIBCUDACXX_COMPILER_MSVC

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
    move_iterator operator++(int) { move_iterator __tmp(*this); ++__current_; return __tmp; }
#endif // _LIBCUDACXX_STD_VER < 17

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
    move_iterator& operator--() { --__current_; return *this; }
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
    move_iterator operator--(int) { move_iterator __tmp(*this); --__current_; return __tmp; }
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
    move_iterator operator+(difference_type __n) const { return move_iterator(__current_ + __n); }
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
    move_iterator& operator+=(difference_type __n) { __current_ += __n; return *this; }
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
    move_iterator operator-(difference_type __n) const { return move_iterator(__current_ - __n); }
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
    move_iterator& operator-=(difference_type __n) { __current_ -= __n; return *this; }

#if _LIBCUDACXX_STD_VER > 14
    _LIBCUDACXX_TEMPLATE(class _Sent)
        (requires sentinel_for<_Sent, _Iter> _LIBCUDACXX_AND __move_iter_comparable<_Iter, _Sent>)
    friend _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
    bool operator==(const move_iterator& __x, const move_sentinel<_Sent>& __y)
    {
        return __x.base() == __y.base();
    }

#if _LIBCUDACXX_STD_VER < 20
    _LIBCUDACXX_TEMPLATE(class _Sent)
        (requires sentinel_for<_Sent, _Iter> _LIBCUDACXX_AND __move_iter_comparable<_Iter, _Sent>)
    friend _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
    bool operator==(const move_sentinel<_Sent>& __y, const move_iterator& __x)
    {
        return __y.base() == __x.base();
    }

    _LIBCUDACXX_TEMPLATE(class _Sent)
        (requires sentinel_for<_Sent, _Iter> _LIBCUDACXX_AND __move_iter_comparable<_Iter, _Sent>)
    friend _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
    bool operator!=(const move_iterator& __x, const move_sentinel<_Sent>& __y)
    {
        return __x.base() != __y.base();
    }

    _LIBCUDACXX_TEMPLATE(class _Sent)
        (requires sentinel_for<_Sent, _Iter> _LIBCUDACXX_AND __move_iter_comparable<_Iter, _Sent>)
    friend _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
    bool operator!=(const move_sentinel<_Sent>& __y, const move_iterator& __x)
    {
        return __y.base() != __x.base();
    }
#endif // _LIBCUDACXX_STD_VER < 20

    _LIBCUDACXX_TEMPLATE(class _Sent)
        (requires sized_sentinel_for<_Sent, _Iter>)
    friend _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
    iter_difference_t<_Iter> operator-(const move_sentinel<_Sent>& __x, const move_iterator& __y)
    {
        return __x.base() - __y.base();
    }

    _LIBCUDACXX_TEMPLATE(class _Sent)
        (requires sized_sentinel_for<_Sent, _Iter>)
    friend _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
    iter_difference_t<_Iter> operator-(const move_iterator& __x, const move_sentinel<_Sent>& __y)
    {
        return __x.base() - __y.base();
    }

    friend _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
    iter_rvalue_reference_t<_Iter> iter_move(const move_iterator& __i)
        noexcept(noexcept(_CUDA_VRANGES::iter_move(_CUDA_VSTD::declval<_Iter>())))
    {
        return _CUDA_VRANGES::iter_move(__i.__current_);
    }

    _LIBCUDACXX_TEMPLATE(class _It2)
        (requires indirectly_swappable<_It2, _Iter>)
    friend _LIBCUDACXX_INLINE_VISIBILITY constexpr
    void iter_swap(const move_iterator& __x, const move_iterator<_It2>& __y)
        noexcept(noexcept(_CUDA_VRANGES::iter_swap(__x.__current_, __y.__current_)))
    {
        return _CUDA_VRANGES::iter_swap(__x.__current_, __y.__current_);
    }
#endif // _LIBCUDACXX_STD_VER > 14
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(move_iterator);

// Some compilers have issues determining _IsFancyPointer
#if defined(_LIBCUDACXX_COMPILER_GCC) \
 || defined(_LIBCUDACXX_COMPILER_MSVC)
template<class _Iter>
struct _IsFancyPointer<move_iterator<_Iter>> : _IsFancyPointer<_Iter> {};
#endif // _LIBCUDACXX_COMPILER_GCC || _LIBCUDACXX_COMPILER_MSVC

template <class _Iter1, class _Iter2>
inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
bool operator==(const move_iterator<_Iter1>& __x, const move_iterator<_Iter2>& __y)
{
    return __x.base() == __y.base();
}

#if _LIBCUDACXX_STD_VER <= 17
template <class _Iter1, class _Iter2>
inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
bool operator!=(const move_iterator<_Iter1>& __x, const move_iterator<_Iter2>& __y)
{
    return __x.base() != __y.base();
}
#endif // _LIBCUDACXX_STD_VER <= 17

#ifndef _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR

template <class _Iter1, three_way_comparable_with<_Iter1> _Iter2>
inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
auto operator<=>(const move_iterator<_Iter1>& __x, const move_iterator<_Iter2>& __y)
    -> compare_three_way_result_t<_Iter1, _Iter2>
{
    return __x.base() <=> __y.base();
}

#else // ^^^ !_LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR ^^^ / vvv _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR vvv
template <class _Iter1, class _Iter2>
inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
bool operator<(const move_iterator<_Iter1>& __x, const move_iterator<_Iter2>& __y)
{
    return __x.base() < __y.base();
}

template <class _Iter1, class _Iter2>
inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
bool operator>(const move_iterator<_Iter1>& __x, const move_iterator<_Iter2>& __y)
{
    return __x.base() > __y.base();
}

template <class _Iter1, class _Iter2>
inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
bool operator<=(const move_iterator<_Iter1>& __x, const move_iterator<_Iter2>& __y)
{
    return __x.base() <= __y.base();
}

template <class _Iter1, class _Iter2>
inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
bool operator>=(const move_iterator<_Iter1>& __x, const move_iterator<_Iter2>& __y)
{
    return __x.base() >= __y.base();
}
#endif // _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR

#ifndef _LIBCUDACXX_CXX03_LANG
template <class _Iter1, class _Iter2>
inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
auto operator-(const move_iterator<_Iter1>& __x, const move_iterator<_Iter2>& __y)
    -> decltype(__x.base() - __y.base())
{
    return __x.base() - __y.base();
}
#else
template <class _Iter1, class _Iter2>
inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
typename move_iterator<_Iter1>::difference_type
operator-(const move_iterator<_Iter1>& __x, const move_iterator<_Iter2>& __y)
{
    return __x.base() - __y.base();
}
#endif // !_LIBCUDACXX_CXX03_LANG

#if _LIBCUDACXX_STD_VER > 17
template <class _Iter>
inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
move_iterator<_Iter> operator+(iter_difference_t<_Iter> __n, const move_iterator<_Iter>& __x)
    requires requires { { __x.base() + __n } -> same_as<_Iter>; }
{
    return __x + __n;
}
#else // ^^^ _LIBCUDACXX_STD_VER > 17 ^^^ / vvv _LIBCUDACXX_STD_VER < 20 vvv
template <class _Iter>
inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
move_iterator<_Iter>
operator+(typename move_iterator<_Iter>::difference_type __n, const move_iterator<_Iter>& __x)
{
    return move_iterator<_Iter>(__x.base() + __n);
}
#endif // _LIBCUDACXX_STD_VER < 20

template <class _Iter>
inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
move_iterator<_Iter>
make_move_iterator(_Iter __i)
{
    return move_iterator<_Iter>(_CUDA_VSTD::move(__i));
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ITERATOR_MOVE_ITERATOR_H
