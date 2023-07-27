//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___UTILITY_PAIR_H
#define _LIBCUDACXX___UTILITY_PAIR_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#ifndef _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR
#include "../__compare/common_comparison_category.h"
#include "../__compare/synth_three_way.h"
#endif // _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR

#include "../__functional/unwrap_ref.h"
#include "../__fwd/get.h"
#include "../__fwd/tuple.h"
#include "../__tuple_dir/sfinae_helpers.h"
#include "../__tuple_dir/structured_bindings.h"
#include "../__tuple_dir/tuple_element.h"
#include "../__tuple_dir/tuple_indices.h"
#include "../__tuple_dir/tuple_size.h"
#include "../__type_traits/common_reference.h"
#include "../__type_traits/conditional.h"
#include "../__type_traits/decay.h"
#include "../__type_traits/enable_if.h"
#include "../__type_traits/integral_constant.h"
#include "../__type_traits/is_assignable.h"
#include "../__type_traits/is_constructible.h"
#include "../__type_traits/is_convertible.h"
#include "../__type_traits/is_copy_assignable.h"
#include "../__type_traits/is_default_constructible.h"
#include "../__type_traits/is_implicitly_default_constructible.h"
#include "../__type_traits/is_move_assignable.h"
#include "../__type_traits/is_nothrow_assignable.h"
#include "../__type_traits/is_nothrow_constructible.h"
#include "../__type_traits/is_nothrow_copy_assignable.h"
#include "../__type_traits/is_nothrow_copy_constructible.h"
#include "../__type_traits/is_nothrow_default_constructible.h"
#include "../__type_traits/is_nothrow_move_assignable.h"
#include "../__type_traits/is_nothrow_move_constructible.h"
#include "../__type_traits/is_same.h"
#include "../__type_traits/is_swappable.h"
#include "../__type_traits/make_const_lvalue_ref.h"
#include "../__utility/forward.h"
#include "../__utility/move.h"
#include "../__utility/piecewise_construct.h"
#include "../cstddef"

// Provide compatability between `std::pair` and `cuda::std::pair`
#if defined(__cuda_std__) && !defined(__CUDACC_RTC__)
#include <utility>
#endif // defined(__cuda_std__) && !defined(__CUDACC_RTC__)

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_LIBCUDACXX_DEPRECATED_ABI_DISABLE_PAIR_TRIVIAL_COPY_CTOR)
template <class, class>
struct __non_trivially_copyable_base {
  _LIBCUDACXX_CONSTEXPR _LIBCUDACXX_INLINE_VISIBILITY
  __non_trivially_copyable_base() _NOEXCEPT {}
  _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 _LIBCUDACXX_INLINE_VISIBILITY
  __non_trivially_copyable_base(__non_trivially_copyable_base const&) _NOEXCEPT {}
};
#endif

template <class _T1, class _T2>
struct _LIBCUDACXX_TEMPLATE_VIS pair
#if defined(_LIBCUDACXX_DEPRECATED_ABI_DISABLE_PAIR_TRIVIAL_COPY_CTOR)
: private __non_trivially_copyable_base<_T1, _T2>
#endif
{
    typedef _T1 first_type;
    typedef _T2 second_type;

    _T1 first;
    _T2 second;

    pair(pair const&) = default;
    pair(pair&&) = default;

    struct _CheckArgs {
      struct __enable_implicit_default : public integral_constant<bool,
                 __is_implicitly_default_constructible<_T1>::value
              && __is_implicitly_default_constructible<_T2>::value>
      {};

      struct __enable_explicit_default : public integral_constant<bool,
                 is_default_constructible<_T1>::value
              && is_default_constructible<_T2>::value
              && !__enable_implicit_default::value>
      {};

      template <class _U1, class _U2>
      struct __is_pair_constructible : public integral_constant<bool,
                 is_constructible<first_type, _U1>::value
              && is_constructible<second_type, _U2>::value>
      {};

      template <class _U1, class _U2>
      struct __is_implicit : public integral_constant<bool,
                 is_convertible<_U1, first_type>::value
              && is_convertible<_U2, second_type>::value>
      {};

      template <class _U1, class _U2>
      struct __enable_explicit : public integral_constant<bool,
                 __is_pair_constructible<_U1, _U2>::value
             && !__is_implicit<_U1, _U2>::value>
      {};

      template <class _U1, class _U2>
      struct __enable_implicit : public integral_constant<bool,
                 __is_pair_constructible<_U1, _U2>::value
             &&  __is_implicit<_U1, _U2>::value>
      {};
    };

    template <bool _MaybeEnable>
    using _CheckArgsDep _LIBCUDACXX_NODEBUG_TYPE = __conditional_t<
      _MaybeEnable, _CheckArgs, __check_tuple_constructor_fail>;

    struct _CheckTupleLikeConstructor {
        template <class _Tuple>
        struct __enable_implicit : public integral_constant<bool,
                    __tuple_convertible<_Tuple, pair>::value>
        {};
        template <class _Tuple>
        struct __enable_explicit : public integral_constant<bool,
                    __tuple_constructible<_Tuple, pair>::value
                && !__tuple_convertible<_Tuple, pair>::value>
        {};
        template <class _Tuple>
        struct __enable_assign : public integral_constant<bool,
                    __tuple_assignable<_Tuple, pair>::value>
        {};
    };

    template <class _Tuple>
    using _CheckTLC _LIBCUDACXX_NODEBUG_TYPE = __conditional_t<
        __tuple_like_with_size<_Tuple, 2>::value
            && !is_same<__decay_t<_Tuple>, pair>::value,
        _CheckTupleLikeConstructor,
        __check_tuple_constructor_fail
    >;

    template<bool _Dummy = true, __enable_if_t<
            _CheckArgsDep<_Dummy>::__enable_explicit_default::value
    >* = nullptr>
    explicit _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR
    pair() _NOEXCEPT_(is_nothrow_default_constructible<first_type>::value &&
                      is_nothrow_default_constructible<second_type>::value)
        : first(), second() {}

    template<bool _Dummy = true, __enable_if_t<
            _CheckArgsDep<_Dummy>::__enable_implicit_default::value
    >* = nullptr>
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR
    pair() _NOEXCEPT_(is_nothrow_default_constructible<first_type>::value &&
                      is_nothrow_default_constructible<second_type>::value)
        : first(), second() {}

    template <bool _Dummy = true, __enable_if_t<
             _CheckArgsDep<_Dummy>::template __enable_explicit<__make_const_lvalue_ref<_T1>, __make_const_lvalue_ref<_T2>>::value
    >* = nullptr>
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
    explicit pair(_T1 const& __t1, _T2 const& __t2)
        _NOEXCEPT_(is_nothrow_copy_constructible<first_type>::value &&
                   is_nothrow_copy_constructible<second_type>::value)
        : first(__t1), second(__t2) {}

    template<bool _Dummy = true, __enable_if_t<
            _CheckArgsDep<_Dummy>::template __enable_implicit<__make_const_lvalue_ref<_T1>, __make_const_lvalue_ref<_T2>>::value
    >* = nullptr>
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
    pair(_T1 const& __t1, _T2 const& __t2)
        _NOEXCEPT_(is_nothrow_copy_constructible<first_type>::value &&
                   is_nothrow_copy_constructible<second_type>::value)
        : first(__t1), second(__t2) {}

    template <
#if _LIBCUDACXX_STD_VER > 20 // http://wg21.link/P1951
        class _U1 = _T1, class _U2 = _T2,
#else
        class _U1, class _U2,
#endif
        __enable_if_t<_CheckArgs::template __enable_explicit<_U1, _U2>::value>* = nullptr
    >
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
    explicit pair(_U1&& __u1, _U2&& __u2)
        _NOEXCEPT_((is_nothrow_constructible<first_type, _U1>::value &&
                    is_nothrow_constructible<second_type, _U2>::value))
        : first(_CUDA_VSTD::forward<_U1>(__u1)), second(_CUDA_VSTD::forward<_U2>(__u2)) {}

    template <
#if _LIBCUDACXX_STD_VER > 20 // http://wg21.link/P1951
        class _U1 = _T1, class _U2 = _T2,
#else
        class _U1, class _U2,
#endif
        __enable_if_t<_CheckArgs::template __enable_implicit<_U1, _U2>::value>* = nullptr
    >
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
    pair(_U1&& __u1, _U2&& __u2)
        _NOEXCEPT_((is_nothrow_constructible<first_type, _U1>::value &&
                    is_nothrow_constructible<second_type, _U2>::value))
        : first(_CUDA_VSTD::forward<_U1>(__u1)), second(_CUDA_VSTD::forward<_U2>(__u2)) {}

#if _LIBCUDACXX_STD_VER > 20
    template<class _U1, class _U2, __enable_if_t<
            _CheckArgs::template __is_pair_constructible<_U1&, _U2&>::value
    >* = nullptr>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
    explicit(!_CheckArgs::template __is_implicit<_U1&, _U2&>()) pair(pair<_U1, _U2>& __p)
        noexcept((is_nothrow_constructible<first_type, _U1&>::value &&
                  is_nothrow_constructible<second_type, _U2&>::value))
        : first(__p.first), second(__p.second) {}

#if defined(__cuda_std__) && !defined(__CUDACC_RTC__)
    template<class _U1, class _U2, __enable_if_t<
            _CheckArgs::template __is_pair_constructible<_U1&, _U2&>::value
    >* = nullptr>
    _LIBCUDACXX_HOST _LIBCUDACXX_INLINE_VISIBILITY constexpr
    explicit(!_CheckArgs::template __is_implicit<_U1&, _U2&>()) pair(::std::pair<_U1, _U2>& __p)
        noexcept((is_nothrow_constructible<first_type, _U1&>::value &&
                  is_nothrow_constructible<second_type, _U2&>::value))
        : first(__p.first), second(__p.second) {}
#endif // defined(__cuda_std__) && !defined(__CUDACC_RTC__)
#endif // _LIBCUDACXX_STD_VER > 20

    template<class _U1, class _U2, __enable_if_t<
            _CheckArgs::template __enable_explicit<_U1 const&, _U2 const&>::value
    >* = nullptr>
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
    explicit pair(pair<_U1, _U2> const& __p)
        _NOEXCEPT_((is_nothrow_constructible<first_type, _U1 const&>::value &&
                    is_nothrow_constructible<second_type, _U2 const&>::value))
        : first(__p.first), second(__p.second) {}

#if defined(__cuda_std__) && !defined(__CUDACC_RTC__)
    template<class _U1, class _U2, __enable_if_t<
            _CheckArgs::template __enable_explicit<_U1 const&, _U2 const&>::value
    >* = nullptr>
    _LIBCUDACXX_HOST _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
    explicit pair(::std::pair<_U1, _U2> const& __p)
        _NOEXCEPT_((is_nothrow_constructible<first_type, _U1 const&>::value &&
                    is_nothrow_constructible<second_type, _U2 const&>::value))
        : first(__p.first), second(__p.second) {}
#endif // defined(__cuda_std__) && !defined(__CUDACC_RTC__)

    template<class _U1, class _U2, __enable_if_t<
            _CheckArgs::template __enable_implicit<_U1 const&, _U2 const&>::value
    >* = nullptr>
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
    pair(pair<_U1, _U2> const& __p)
        _NOEXCEPT_((is_nothrow_constructible<first_type, _U1 const&>::value &&
                    is_nothrow_constructible<second_type, _U2 const&>::value))
        : first(__p.first), second(__p.second) {}

#if defined(__cuda_std__) && !defined(__CUDACC_RTC__)
    template<class _U1, class _U2, __enable_if_t<
            _CheckArgs::template __enable_implicit<_U1 const&, _U2 const&>::value
    >* = nullptr>
    _LIBCUDACXX_HOST _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
    pair(::std::pair<_U1, _U2> const& __p)
        _NOEXCEPT_((is_nothrow_constructible<first_type, _U1 const&>::value &&
                    is_nothrow_constructible<second_type, _U2 const&>::value))
        : first(__p.first), second(__p.second) {}
#endif // defined(__cuda_std__) && !defined(__CUDACC_RTC__)

    template<class _U1, class _U2, __enable_if_t<
            _CheckArgs::template __enable_explicit<_U1, _U2>::value
    >* = nullptr>
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
    explicit pair(pair<_U1, _U2>&&__p)
        _NOEXCEPT_((is_nothrow_constructible<first_type, _U1&&>::value &&
                    is_nothrow_constructible<second_type, _U2&&>::value))
        : first(_CUDA_VSTD::forward<_U1>(__p.first)), second(_CUDA_VSTD::forward<_U2>(__p.second)) {}

#if defined(__cuda_std__) && !defined(__CUDACC_RTC__)
    template<class _U1, class _U2, __enable_if_t<
            _CheckArgs::template __enable_explicit<_U1, _U2>::value
    >* = nullptr>
    _LIBCUDACXX_HOST _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
    explicit pair(::std::pair<_U1, _U2>&&__p)
        _NOEXCEPT_((is_nothrow_constructible<first_type, _U1&&>::value &&
                    is_nothrow_constructible<second_type, _U2&&>::value))
        : first(_CUDA_VSTD::forward<_U1>(__p.first)), second(_CUDA_VSTD::forward<_U2>(__p.second)) {}
#endif // defined(__cuda_std__) && !defined(__CUDACC_RTC__)

    template<class _U1, class _U2, __enable_if_t<
            _CheckArgs::template __enable_implicit<_U1, _U2>::value
    >* = nullptr>
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
    pair(pair<_U1, _U2>&& __p)
        _NOEXCEPT_((is_nothrow_constructible<first_type, _U1&&>::value &&
                    is_nothrow_constructible<second_type, _U2&&>::value))
        : first(_CUDA_VSTD::forward<_U1>(__p.first)), second(_CUDA_VSTD::forward<_U2>(__p.second)) {}

#if defined(__cuda_std__) && !defined(__CUDACC_RTC__)
    template<class _U1, class _U2, __enable_if_t<
            _CheckArgs::template __enable_implicit<_U1, _U2>::value
    >* = nullptr>
    _LIBCUDACXX_HOST _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
    pair(::std::pair<_U1, _U2>&& __p)
        _NOEXCEPT_((is_nothrow_constructible<first_type, _U1&&>::value &&
                    is_nothrow_constructible<second_type, _U2&&>::value))
        : first(_CUDA_VSTD::forward<_U1>(__p.first)), second(_CUDA_VSTD::forward<_U2>(__p.second)) {}
#endif // defined(__cuda_std__) && !defined(__CUDACC_RTC__)

#if _LIBCUDACXX_STD_VER > 20
    template<class _U1, class _U2, __enable_if_t<
            _CheckArgs::template __is_pair_constructible<const _U1&&, const _U2&&>::value
    >* = nullptr>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
    explicit(!_CheckArgs::template __is_implicit<const _U1&&, const _U2&&>::value)
    pair(const pair<_U1, _U2>&& __p)
        noexcept(is_nothrow_constructible<first_type, const _U1&&>::value &&
                 is_nothrow_constructible<second_type, const _U2&&>::value)
        : first(_CUDA_VSTD::move(__p.first)), second(_CUDA_VSTD::move(__p.second)) {}

#if defined(__cuda_std__) && !defined(__CUDACC_RTC__)
    template<class _U1, class _U2, __enable_if_t<
            _CheckArgs::template __is_pair_constructible<const _U1&&, const _U2&&>::value
    >* = nullptr>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_HOST constexpr
    explicit(!_CheckArgs::template __is_implicit<const _U1&&, const _U2&&>::value)
    pair(const ::std::pair<_U1, _U2>&& __p)
        noexcept(is_nothrow_constructible<first_type, const _U1&&>::value &&
                 is_nothrow_constructible<second_type, const _U2&&>::value)
        : first(_CUDA_VSTD::move(__p.first)), second(_CUDA_VSTD::move(__p.second)) {}
#endif // defined(__cuda_std__) && !defined(__CUDACC_RTC__)
#endif // _LIBCUDACXX_STD_VER > 20

    template<class _Tuple, __enable_if_t<
            _CheckTLC<_Tuple>::template __enable_explicit<_Tuple>::value
    >* = nullptr>
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
    explicit pair(_Tuple&& __p)
        : first(_CUDA_VSTD::get<0>(_CUDA_VSTD::forward<_Tuple>(__p))),
          second(_CUDA_VSTD::get<1>(_CUDA_VSTD::forward<_Tuple>(__p))) {}

    template<class _Tuple, __enable_if_t<
            _CheckTLC<_Tuple>::template __enable_implicit<_Tuple>::value
    >* = nullptr>
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
    pair(_Tuple&& __p)
        : first(_CUDA_VSTD::get<0>(_CUDA_VSTD::forward<_Tuple>(__p))),
          second(_CUDA_VSTD::get<1>(_CUDA_VSTD::forward<_Tuple>(__p))) {}

    template <class... _Args1, class... _Args2>
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
    pair(piecewise_construct_t __pc,
         tuple<_Args1...> __first_args, tuple<_Args2...> __second_args)
        _NOEXCEPT_((is_nothrow_constructible<first_type, _Args1...>::value &&
                    is_nothrow_constructible<second_type, _Args2...>::value))
        : pair(__pc, __first_args, __second_args,
                typename __make_tuple_indices<sizeof...(_Args1)>::type(),
                typename __make_tuple_indices<sizeof...(_Args2) >::type()) {}

    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
    pair& operator=(__conditional_t<
                        is_copy_assignable<first_type>::value &&
                        is_copy_assignable<second_type>::value,
                    pair, __nat> const& __p)
        _NOEXCEPT_(is_nothrow_copy_assignable<first_type>::value &&
                   is_nothrow_copy_assignable<second_type>::value)
    {
        first = __p.first;
        second = __p.second;
        return *this;
    }

#if defined(__cuda_std__) && !defined(__CUDACC_RTC__)
    template<class _UT1 = _T1, __enable_if_t<is_copy_assignable<_UT1>::value &&
                                             is_copy_assignable<_T2>::value, int> = 0>
    _LIBCUDACXX_HOST _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
    pair& operator=(::std::pair<_T1, _T2> const& __p)
        _NOEXCEPT_(is_nothrow_copy_assignable<first_type>::value &&
                   is_nothrow_copy_assignable<second_type>::value)
    {
        first = __p.first;
        second = __p.second;
        return *this;
    }
#endif // defined(__cuda_std__) && !defined(__CUDACC_RTC__)

    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
    pair& operator=(__conditional_t<
                        is_move_assignable<first_type>::value &&
                        is_move_assignable<second_type>::value,
                    pair, __nat>&& __p)
        _NOEXCEPT_(is_nothrow_move_assignable<first_type>::value &&
                   is_nothrow_move_assignable<second_type>::value)
    {
        first = _CUDA_VSTD::forward<first_type>(__p.first);
        second = _CUDA_VSTD::forward<second_type>(__p.second);
        return *this;
    }

#if defined(__cuda_std__) && !defined(__CUDACC_RTC__)
    template<class _UT1 = _T1, __enable_if_t<is_move_assignable<_UT1>::value &&
                                             is_move_assignable<_T2>::value, int> = 0>
    _LIBCUDACXX_HOST _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
    pair& operator=(::std::pair<_T1, _T2>&& __p)
        _NOEXCEPT_(is_nothrow_move_assignable<first_type>::value &&
                   is_nothrow_move_assignable<second_type>::value)
    {
        first = _CUDA_VSTD::forward<first_type>(__p.first);
        second = _CUDA_VSTD::forward<second_type>(__p.second);
        return *this;
    }
#endif // defined(__cuda_std__) && !defined(__CUDACC_RTC__)

#if _LIBCUDACXX_STD_VER > 20
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
    const pair& operator=(pair const& __p) const
      noexcept(is_nothrow_copy_assignable_v<const first_type> &&
               is_nothrow_copy_assignable_v<const second_type>)
      requires(is_copy_assignable_v<const first_type> &&
               is_copy_assignable_v<const second_type>) {
        first = __p.first;
        second = __p.second;
        return *this;
    }

#if defined(__cuda_std__) && !defined(__CUDACC_RTC__)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_HOST constexpr
    const pair& operator=(::std::pair<_T1, _T2> const& __p) const
      noexcept(is_nothrow_copy_assignable_v<const first_type> &&
               is_nothrow_copy_assignable_v<const second_type>)
      requires(is_copy_assignable_v<const first_type> &&
               is_copy_assignable_v<const second_type>) {
        first = __p.first;
        second = __p.second;
        return *this;
    }
#endif // defined(__cuda_std__) && !defined(__CUDACC_RTC__)

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
    const pair& operator=(pair&& __p) const
      noexcept(is_nothrow_assignable_v<const first_type&, first_type> &&
               is_nothrow_assignable_v<const second_type&, second_type>)
      requires(is_assignable_v<const first_type&, first_type> &&
               is_assignable_v<const second_type&, second_type>) {
        first = _CUDA_VSTD::forward<first_type>(__p.first);
        second = _CUDA_VSTD::forward<second_type>(__p.second);
        return *this;
    }

#if defined(__cuda_std__) && !defined(__CUDACC_RTC__)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_HOST constexpr
    const pair& operator=(::std::pair<_T1, _T2>&& __p) const
      noexcept(is_nothrow_assignable_v<const first_type&, first_type> &&
               is_nothrow_assignable_v<const second_type&, second_type>)
      requires(is_assignable_v<const first_type&, first_type> &&
               is_assignable_v<const second_type&, second_type>) {
        first = _CUDA_VSTD::forward<first_type>(__p.first);
        second = _CUDA_VSTD::forward<second_type>(__p.second);
        return *this;
    }
#endif // defined(__cuda_std__) && !defined(__CUDACC_RTC__)

    template<class _U1, class _U2>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
    const pair& operator=(const pair<_U1, _U2>& __p) const
      requires(is_assignable_v<const first_type&, const _U1&> &&
               is_assignable_v<const second_type&, const _U2&>) {
        first = __p.first;
        second = __p.second;
        return *this;
    }

#if defined(__cuda_std__) && !defined(__CUDACC_RTC__)
    template<class _U1, class _U2>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_HOST constexpr
    const pair& operator=(const ::std::pair<_U1, _U2>& __p) const
      requires(is_assignable_v<const first_type&, const _U1&> &&
               is_assignable_v<const second_type&, const _U2&>) {
        first = __p.first;
        second = __p.second;
        return *this;
    }
#endif // defined(__cuda_std__) && !defined(__CUDACC_RTC__)

    template<class _U1, class _U2>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
    const pair& operator=(pair<_U1, _U2>&& __p) const
      requires(is_assignable_v<const first_type&, _U1> &&
               is_assignable_v<const second_type&, _U2>) {
        first = _CUDA_VSTD::forward<_U1>(__p.first);
        second = _CUDA_VSTD::forward<_U2>(__p.second);
        return *this;
    }

#if defined(__cuda_std__) && !defined(__CUDACC_RTC__)
    template<class _U1, class _U2>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_HOST constexpr
    const pair& operator=(::std::pair<_U1, _U2>&& __p) const
      requires(is_assignable_v<const first_type&, _U1> &&
               is_assignable_v<const second_type&, _U2>) {
        first = _CUDA_VSTD::forward<_U1>(__p.first);
        second = _CUDA_VSTD::forward<_U2>(__p.second);
        return *this;
    }
#endif // defined(__cuda_std__) && !defined(__CUDACC_RTC__)
#endif // _LIBCUDACXX_STD_VER > 20

    template <class _Tuple, __enable_if_t<
            _CheckTLC<_Tuple>::template __enable_assign<_Tuple>::value
     >* = nullptr>
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
    pair& operator=(_Tuple&& __p) {
        first = _CUDA_VSTD::get<0>(_CUDA_VSTD::forward<_Tuple>(__p));
        second = _CUDA_VSTD::get<1>(_CUDA_VSTD::forward<_Tuple>(__p));
        return *this;
    }

    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
    void swap(pair& __p) _NOEXCEPT_(__is_nothrow_swappable<first_type>::value &&
                                    __is_nothrow_swappable<second_type>::value)
    {
        using _CUDA_VSTD::swap;
        swap(first,  __p.first);
        swap(second, __p.second);
    }

#if _LIBCUDACXX_STD_VER > 20
    _LIBCUDACXX_HIDE_FROM_ABI constexpr
    void swap(const pair& __p) const
        noexcept(__is_nothrow_swappable<const first_type>::value &&
                 __is_nothrow_swappable<const second_type>::value)
    {
        using _CUDA_VSTD::swap;
        swap(first,  __p.first);
        swap(second, __p.second);
    }
#endif // _LIBCUDACXX_STD_VER > 20

#if defined(__cuda_std__) && !defined(__CUDACC_RTC__)
    _LIBCUDACXX_HOST _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
    operator ::std::pair<_T1, _T2>() const { return { first, second }; }
#endif // defined(__cuda_std__) && !defined(__CUDACC_RTC__)

private:

    template <class... _Args1, class... _Args2, size_t... _I1, size_t... _I2>
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
    pair(piecewise_construct_t,
         tuple<_Args1...>& __first_args, tuple<_Args2...>& __second_args,
         __tuple_indices<_I1...>, __tuple_indices<_I2...>);
};

#if _LIBCUDACXX_STD_VER > 14 && !defined(_LIBCUDACXX_HAS_NO_DEDUCTION_GUIDES)
template<class _T1, class _T2>
pair(_T1, _T2) -> pair<_T1, _T2>;
#endif // _LIBCUDACXX_STD_VER > 14 && !defined(_LIBCUDACXX_HAS_NO_DEDUCTION_GUIDES)

// [pairs.spec], specialized algorithms

template <class _T1, class _T2>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
bool
operator==(const pair<_T1,_T2>& __x, const pair<_T1,_T2>& __y)
{
    return __x.first == __y.first && __x.second == __y.second;
}

#ifndef _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR

template <class _T1, class _T2>
_LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
common_comparison_category_t<
        __synth_three_way_result<_T1>,
        __synth_three_way_result<_T2> >
operator<=>(const pair<_T1,_T2>& __x, const pair<_T1,_T2>& __y)
{
    if (auto __c = _CUDA_VSTD::__synth_three_way(__x.first, __y.first); __c != 0) {
      return __c;
    }
    return _CUDA_VSTD::__synth_three_way(__x.second, __y.second);
}

#else // _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR

template <class _T1, class _T2>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
bool
operator!=(const pair<_T1,_T2>& __x, const pair<_T1,_T2>& __y)
{
    return !(__x == __y);
}

template <class _T1, class _T2>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
bool
operator< (const pair<_T1,_T2>& __x, const pair<_T1,_T2>& __y)
{
    return __x.first < __y.first || (!(__y.first < __x.first) && __x.second < __y.second);
}

template <class _T1, class _T2>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
bool
operator> (const pair<_T1,_T2>& __x, const pair<_T1,_T2>& __y)
{
    return __y < __x;
}

template <class _T1, class _T2>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
bool
operator>=(const pair<_T1,_T2>& __x, const pair<_T1,_T2>& __y)
{
    return !(__x < __y);
}

template <class _T1, class _T2>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
bool
operator<=(const pair<_T1,_T2>& __x, const pair<_T1,_T2>& __y)
{
    return !(__y < __x);
}

#endif // _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR

#if _LIBCUDACXX_STD_VER > 17
template <class _T1, class _T2, class _U1, class _U2, template<class> class _TQual, template<class> class _UQual>
    requires requires { typename pair<common_reference_t<_TQual<_T1>, _UQual<_U1>>,
                                      common_reference_t<_TQual<_T2>, _UQual<_U2>>>; }
struct basic_common_reference<pair<_T1, _T2>, pair<_U1, _U2>, _TQual, _UQual> {
    using type = pair<common_reference_t<_TQual<_T1>, _UQual<_U1>>,
                      common_reference_t<_TQual<_T2>, _UQual<_U2>>>;
};

template <class _T1, class _T2, class _U1, class _U2>
    requires requires { typename pair<common_type_t<_T1, _U1>, common_type_t<_T2, _U2>>; }
struct common_type<pair<_T1, _T2>, pair<_U1, _U2>> {
    using type = pair<common_type_t<_T1, _U1>, common_type_t<_T2, _U2>>;
};
#endif // _LIBCUDACXX_STD_VER > 17

template <class _T1, class _T2>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
__enable_if_t
<
    __is_swappable<_T1>::value &&
    __is_swappable<_T2>::value,
    void
>
swap(pair<_T1, _T2>& __x, pair<_T1, _T2>& __y)
                     _NOEXCEPT_((__is_nothrow_swappable<_T1>::value &&
                                 __is_nothrow_swappable<_T2>::value))
{
    __x.swap(__y);
}

#if _LIBCUDACXX_STD_VER > 20
template <class _T1, class _T2>
  requires (__is_swappable<const _T1>::value &&
            __is_swappable<const _T2>::value)
_LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
void swap(const pair<_T1, _T2>& __x, const pair<_T1, _T2>& __y)
    noexcept(noexcept(__x.swap(__y)))
{
    __x.swap(__y);
}
#endif // _LIBCUDACXX_STD_VER > 20

template <class _T1, class _T2>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
pair<typename __unwrap_ref_decay<_T1>::type, typename __unwrap_ref_decay<_T2>::type>
make_pair(_T1&& __t1, _T2&& __t2)
{
    return pair<typename __unwrap_ref_decay<_T1>::type, typename __unwrap_ref_decay<_T2>::type>
               (_CUDA_VSTD::forward<_T1>(__t1), _CUDA_VSTD::forward<_T2>(__t2));
}

template <class _T1, class _T2>
  struct _LIBCUDACXX_TEMPLATE_VIS tuple_size<pair<_T1, _T2> >
    : public integral_constant<size_t, 2> {};

template <size_t _Ip, class _T1, class _T2>
struct _LIBCUDACXX_TEMPLATE_VIS tuple_element<_Ip, pair<_T1, _T2> >
{
    static_assert(_Ip < 2, "Index out of bounds in std::tuple_element<std::pair<T1, T2>>");
};

template <class _T1, class _T2>
struct _LIBCUDACXX_TEMPLATE_VIS tuple_element<0, pair<_T1, _T2> >
{
    typedef _LIBCUDACXX_NODEBUG_TYPE _T1 type;
};

template <class _T1, class _T2>
struct _LIBCUDACXX_TEMPLATE_VIS tuple_element<1, pair<_T1, _T2> >
{
    typedef _LIBCUDACXX_NODEBUG_TYPE _T2 type;
};

template <size_t _Ip> struct __get_pair;

template <>
struct __get_pair<0>
{
    template <class _T1, class _T2>
    static
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
    _T1&
    get(pair<_T1, _T2>& __p) _NOEXCEPT {return __p.first;}

    template <class _T1, class _T2>
    static
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
    const _T1&
    get(const pair<_T1, _T2>& __p) _NOEXCEPT {return __p.first;}

    template <class _T1, class _T2>
    static
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
    _T1&&
    get(pair<_T1, _T2>&& __p) _NOEXCEPT {return _CUDA_VSTD::forward<_T1>(__p.first);}

    template <class _T1, class _T2>
    static
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
    const _T1&&
    get(const pair<_T1, _T2>&& __p) _NOEXCEPT {return _CUDA_VSTD::forward<const _T1>(__p.first);}
};

template <>
struct __get_pair<1>
{
    template <class _T1, class _T2>
    static
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
    _T2&
    get(pair<_T1, _T2>& __p) _NOEXCEPT {return __p.second;}

    template <class _T1, class _T2>
    static
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
    const _T2&
    get(const pair<_T1, _T2>& __p) _NOEXCEPT {return __p.second;}

    template <class _T1, class _T2>
    static
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
    _T2&&
    get(pair<_T1, _T2>&& __p) _NOEXCEPT {return _CUDA_VSTD::forward<_T2>(__p.second);}

    template <class _T1, class _T2>
    static
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
    const _T2&&
    get(const pair<_T1, _T2>&& __p) _NOEXCEPT {return _CUDA_VSTD::forward<const _T2>(__p.second);}
};

template <size_t _Ip, class _T1, class _T2>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
typename tuple_element<_Ip, pair<_T1, _T2> >::type&
get(pair<_T1, _T2>& __p) _NOEXCEPT
{
    return __get_pair<_Ip>::get(__p);
}

template <size_t _Ip, class _T1, class _T2>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
const typename tuple_element<_Ip, pair<_T1, _T2> >::type&
get(const pair<_T1, _T2>& __p) _NOEXCEPT
{
    return __get_pair<_Ip>::get(__p);
}

template <size_t _Ip, class _T1, class _T2>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
typename tuple_element<_Ip, pair<_T1, _T2> >::type&&
get(pair<_T1, _T2>&& __p) _NOEXCEPT
{
    return __get_pair<_Ip>::get(_CUDA_VSTD::move(__p));
}

template <size_t _Ip, class _T1, class _T2>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
const typename tuple_element<_Ip, pair<_T1, _T2> >::type&&
get(const pair<_T1, _T2>&& __p) _NOEXCEPT
{
    return __get_pair<_Ip>::get(_CUDA_VSTD::move(__p));
}

#if _LIBCUDACXX_STD_VER > 11
template <class _T1, class _T2>
inline _LIBCUDACXX_INLINE_VISIBILITY
constexpr _T1 & get(pair<_T1, _T2>& __p) _NOEXCEPT
{
    return __get_pair<0>::get(__p);
}

template <class _T1, class _T2>
inline _LIBCUDACXX_INLINE_VISIBILITY
constexpr _T1 const & get(pair<_T1, _T2> const& __p) _NOEXCEPT
{
    return __get_pair<0>::get(__p);
}

template <class _T1, class _T2>
inline _LIBCUDACXX_INLINE_VISIBILITY
constexpr _T1 && get(pair<_T1, _T2>&& __p) _NOEXCEPT
{
    return __get_pair<0>::get(_CUDA_VSTD::move(__p));
}

template <class _T1, class _T2>
inline _LIBCUDACXX_INLINE_VISIBILITY
constexpr _T1 const && get(pair<_T1, _T2> const&& __p) _NOEXCEPT
{
    return __get_pair<0>::get(_CUDA_VSTD::move(__p));
}

template <class _T1, class _T2>
inline _LIBCUDACXX_INLINE_VISIBILITY
constexpr _T1 & get(pair<_T2, _T1>& __p) _NOEXCEPT
{
    return __get_pair<1>::get(__p);
}

template <class _T1, class _T2>
inline _LIBCUDACXX_INLINE_VISIBILITY
constexpr _T1 const & get(pair<_T2, _T1> const& __p) _NOEXCEPT
{
    return __get_pair<1>::get(__p);
}

template <class _T1, class _T2>
inline _LIBCUDACXX_INLINE_VISIBILITY
constexpr _T1 && get(pair<_T2, _T1>&& __p) _NOEXCEPT
{
    return __get_pair<1>::get(_CUDA_VSTD::move(__p));
}

template <class _T1, class _T2>
inline _LIBCUDACXX_INLINE_VISIBILITY
constexpr _T1 const && get(pair<_T2, _T1> const&& __p) _NOEXCEPT
{
    return __get_pair<1>::get(_CUDA_VSTD::move(__p));
}

#endif // _LIBCUDACXX_STD_VER > 11

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___UTILITY_PAIR_H
