//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___UTILITY_PAIR_H
#define _CUDA_STD___UTILITY_PAIR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#  include <cuda/std/__compare/common_comparison_category.h>
#  include <cuda/std/__compare/synth_three_way.h>
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__functional/unwrap_ref.h>
#include <cuda/std/__fwd/get.h>
#include <cuda/std/__fwd/pair.h>
#include <cuda/std/__fwd/subrange.h>
#include <cuda/std/__fwd/tuple.h>
#include <cuda/std/__tuple_dir/sfinae_helpers.h>
#include <cuda/std/__tuple_dir/tuple_constraints.h>
#include <cuda/std/__tuple_dir/tuple_element.h>
#include <cuda/std/__tuple_dir/tuple_indices.h>
#include <cuda/std/__tuple_dir/tuple_like.h>
#include <cuda/std/__tuple_dir/tuple_size.h>
#include <cuda/std/__type_traits/common_reference.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_assignable.h>
#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_copy_assignable.h>
#include <cuda/std/__type_traits/is_default_constructible.h>
#include <cuda/std/__type_traits/is_implicitly_default_constructible.h>
#include <cuda/std/__type_traits/is_move_assignable.h>
#include <cuda/std/__type_traits/is_nothrow_assignable.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_copy_assignable.h>
#include <cuda/std/__type_traits/is_nothrow_copy_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_default_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_move_assignable.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_swappable.h>
#include <cuda/std/__type_traits/make_const_lvalue_ref.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/piecewise_construct.h>

#include <cuda/std/__cccl/prologue.h>

// On Windows we are getting a warning when compiling the const assignment operators with a reference type
_CCCL_BEGIN_NV_DIAG_SUPPRESS(1770) // type qualifiers are ignored (underlying type is a reference)

_CCCL_BEGIN_NAMESPACE_CUDA_STD

_CCCL_EXEC_CHECK_DISABLE
template <class _UPair, class _T1, class _T2>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL __select_constructor __pair_select_pair_like_constructible() noexcept
{
  using ::cuda::std::get;
  if constexpr (__is_cuda_std_ranges_subrange_v<remove_cvref_t<_UPair>>)
  { // [pairs#pair]-15.1: remove_cvref_t<UTuple> is not a specialization of ranges​::​subrange,
    return __select_constructor::__none;
  }
  else if constexpr (!__pair_like<_UPair>)
  {
    return __select_constructor::__none;
  }
  else if constexpr (!is_constructible_v<_T1, decltype(get<0>(::cuda::std::declval<_UPair>()))>)
  { // [pairs#pair]-15.2: is_constructible<T1, decltype(get<0>(std​::​forward<UTuple>(u)))>... is true
    return __select_constructor::__none;
  }
  else if constexpr (!is_constructible_v<_T2, decltype(get<1>(::cuda::std::declval<_UPair>()))>)
  { // [pairs#pair]-15.3: is_constructible<T2, decltype(get<1>(std​::​forward<UTuple>(u)))>... is true
    return __select_constructor::__none;
  }
  else if constexpr (is_convertible_v<decltype(get<0>(::cuda::std::declval<_UPair>())), _T1>
                     && is_convertible_v<decltype(get<1>(::cuda::std::declval<_UPair>())), _T2>)
  { // [pairs#pair]-17 !is_convertible_v<decltype(get<0>(FWD(u))), T1> &&
    //                 !is_convertible_v<decltype(get<1>(FWD(u))), T2>
    return __select_constructor::__implicit;
  }
  else
  {
    return __select_constructor::__explicit;
  }
}

template <class _UPair, class _T1, class _T2>
inline constexpr __select_constructor __pair_select_pair_like_constructible_v =
  ::cuda::std::__pair_select_pair_like_constructible<_UPair, _T1, _T2>();

_CCCL_EXEC_CHECK_DISABLE
template <bool _IsConst, class _UPair, class _T1, class _T2>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL __select_assignment __pair_select_pair_like_assignable() noexcept
{
  using ::cuda::std::get;
  if constexpr (is_same_v<remove_cvref_t<_UPair>, pair<_T1, _T2>>)
  { // [pairs.pair]-42.1: different-from<UPair, pair>
    // [pairs.pair]-45.1: different-from<UPair, pair>
    return __select_assignment::__none;
  }
  else if constexpr (__is_cuda_std_ranges_subrange_v<remove_cvref_t<_UPair>>)
  { // [pairs.pair]-42.2: remove_cvref_t<UTuple> is not a specialization of ranges​::​subrange,
    // [pairs.pair]-45.2: remove_cvref_t<UTuple> is not a specialization of ranges​::​subrange,
    return __select_assignment::__none;
  }
  else if constexpr (!__pair_like<_UPair>)
  { // [pairs.pair]-42: pair-like P
    // [pairs.pair]-45: pair-like P
    return __select_assignment::__none;
  }
  else if constexpr (_IsConst)
  {
    if constexpr (!is_assignable_v<const _T1&, decltype(get<0>(::cuda::std::declval<_UPair>()))>)
    { // [pairs.pair]-45.3: is_assignable_v<const T1&, decltype(get<0>(std​::​forward<P>(p)))> is true
      return __select_assignment::__none;
    }
    else if constexpr (!is_assignable_v<const _T2&, decltype(get<1>(::cuda::std::declval<_UPair>()))>)
    { // [pairs.pair]-45.4: is_assignable_v<const T2&, decltype(get<1>(std​::​forward<P>(p)))> is true
      return __select_assignment::__none;
    }
    else if constexpr (is_nothrow_assignable_v<const _T1&, decltype(get<0>(::cuda::std::declval<_UPair>()))>
                       && is_nothrow_assignable_v<const _T2&, decltype(get<1>(::cuda::std::declval<_UPair>()))>)
    {
      return __select_assignment::__is_nothrow;
    }
    else
    {
      return __select_assignment::__may_throw;
    }
  }
  else if constexpr (!is_assignable_v<_T1&, decltype(get<0>(::cuda::std::declval<_UPair>()))>)
  { // [pairs.pair]-42.3: is_assignable_v<const T1&, decltype(get<0>(std​::​forward<P>(p)))> is true
    return __select_assignment::__none;
  }
  else if constexpr (!is_assignable_v<_T2&, decltype(get<1>(::cuda::std::declval<_UPair>()))>)
  { // [pairs.pair]-42.4: is_assignable_v<const T2&, decltype(get<1>(std​::​forward<P>(p)))> is true
    return __select_assignment::__none;
  }
  else if constexpr (is_nothrow_assignable_v<_T1&, decltype(get<0>(::cuda::std::declval<_UPair>()))>
                     && is_nothrow_assignable_v<_T2&, decltype(get<1>(::cuda::std::declval<_UPair>()))>)
  {
    return __select_assignment::__is_nothrow;
  }
  else
  {
    return __select_assignment::__may_throw;
  }
}

template <bool _IsConst, class _UPair, class _T1, class _T2>
inline constexpr __select_assignment __pair_select_pair_like_assignable_v =
  ::cuda::std::__pair_select_pair_like_assignable<_IsConst, _UPair, _T1, _T2>();

// base class to ensure `is_trivially_copyable` when possible
template <class _T1, class _T2, bool = __must_synthesize_assignment_v<_T1> || __must_synthesize_assignment_v<_T2>>
struct __pair_base
{
  _T1 first;
  _T2 second;

  _CCCL_EXEC_CHECK_DISABLE
  template <__select_constructor _Trait = ::cuda::std::__tuple_select_default_constructible(__tuple_types<_T1, _T2>{}),
            enable_if_t<__can_construct_implicitly<_Trait>, int> = 0>
  _CCCL_API constexpr __pair_base() noexcept(is_nothrow_default_constructible_v<_T1>
                                             && is_nothrow_default_constructible_v<_T2>)
      : first()
      , second()
  {}

  _CCCL_EXEC_CHECK_DISABLE
  template <__select_constructor _Trait = ::cuda::std::__tuple_select_default_constructible(__tuple_types<_T1, _T2>{}),
            enable_if_t<__can_construct_explicitly<_Trait>, int> = 0>
  _CCCL_API explicit constexpr __pair_base() noexcept(
    is_nothrow_default_constructible_v<_T1> && is_nothrow_default_constructible_v<_T2>)
      : first()
      , second()
  {}

  _CCCL_EXEC_CHECK_DISABLE
  template <class _U1, class _U2>
  _CCCL_API constexpr __pair_base(_U1&& __t1, _U2&& __t2) noexcept(
    is_nothrow_constructible_v<_T1, _U1> && is_nothrow_constructible_v<_T2, _U2>)
      : first(::cuda::std::forward<_U1>(__t1))
      , second(::cuda::std::forward<_U2>(__t2))
  {}

protected:
  template <class... _Args1, class... _Args2, size_t... _I1, size_t... _I2>
  _CCCL_API constexpr __pair_base(
    piecewise_construct_t,
    tuple<_Args1...>& __first_args,
    tuple<_Args2...>& __second_args,
    __tuple_indices<_I1...>,
    __tuple_indices<_I2...>);
};

template <class _T1, class _T2>
struct __pair_base<_T1, _T2, true>
{
  _T1 first;
  _T2 second;

  _CCCL_EXEC_CHECK_DISABLE
  template <__select_constructor _Trait = ::cuda::std::__tuple_select_default_constructible(__tuple_types<_T1, _T2>{}),
            enable_if_t<__can_construct_implicitly<_Trait>, int> = 0>
  _CCCL_API constexpr __pair_base() noexcept(is_nothrow_default_constructible_v<_T1>
                                             && is_nothrow_default_constructible_v<_T2>)
      : first()
      , second()
  {}

  _CCCL_EXEC_CHECK_DISABLE
  template <__select_constructor _Trait = ::cuda::std::__tuple_select_default_constructible(__tuple_types<_T1, _T2>{}),
            enable_if_t<__can_construct_explicitly<_Trait>, int> = 0>
  _CCCL_API explicit constexpr __pair_base() noexcept(
    is_nothrow_default_constructible_v<_T1> && is_nothrow_default_constructible_v<_T2>)
      : first()
      , second()
  {}

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HIDE_FROM_ABI constexpr __pair_base(const __pair_base&) = default;
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HIDE_FROM_ABI constexpr __pair_base(__pair_base&&) = default;

  // We need to ensure that a reference type, which would inhibit the implicit copy assignment still works
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr __pair_base&
  operator=(const conditional_t<is_copy_assignable_v<_T1> && is_copy_assignable_v<_T2>, __pair_base, __nat>&
              __p) noexcept(is_nothrow_copy_assignable_v<_T1> && is_nothrow_copy_assignable_v<_T2>)
  {
    first  = __p.first;
    second = __p.second;
    return *this;
  }

  // We need to ensure that a reference type, which would inhibit the implicit move assignment still works
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr __pair_base&
  operator=(conditional_t<is_move_assignable_v<_T1> && is_move_assignable_v<_T2>, __pair_base, __nat>&& __p) noexcept(
    is_nothrow_move_assignable_v<_T1> && is_nothrow_move_assignable_v<_T2>)
  {
    first  = ::cuda::std::move(__p.first);
    second = ::cuda::std::move(__p.second);
    return *this;
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <class _U1, class _U2>
  _CCCL_API constexpr __pair_base(_U1&& __t1, _U2&& __t2) noexcept(
    is_nothrow_constructible_v<_T1, _U1> && is_nothrow_constructible_v<_T2, _U2>)
      : first(::cuda::std::forward<_U1>(__t1))
      , second(::cuda::std::forward<_U2>(__t2))
  {}

protected:
  template <class... _Args1, class... _Args2, size_t... _I1, size_t... _I2>
  _CCCL_API constexpr __pair_base(
    piecewise_construct_t,
    tuple<_Args1...>& __first_args,
    tuple<_Args2...>& __second_args,
    __tuple_indices<_I1...>,
    __tuple_indices<_I2...>);
};

template <class _T1, class _T2>
struct _CCCL_TYPE_VISIBILITY_DEFAULT pair : public __pair_base<_T1, _T2>
{
  using __base = __pair_base<_T1, _T2>;

  using first_type  = _T1;
  using second_type = _T2;

  template <__select_constructor _Trait = ::cuda::std::__tuple_select_default_constructible(__tuple_types<_T1, _T2>{}),
            enable_if_t<__can_construct_implicitly<_Trait>, int> = 0>
  _CCCL_API constexpr pair() noexcept(is_nothrow_default_constructible_v<_T1> && is_nothrow_default_constructible_v<_T2>)
      : __base()
  {}

  template <__select_constructor _Trait = ::cuda::std::__tuple_select_default_constructible(__tuple_types<_T1, _T2>{}),
            enable_if_t<__can_construct_explicitly<_Trait>, int> = 0>
  _CCCL_API explicit constexpr pair() noexcept(is_nothrow_default_constructible_v<_T1>
                                               && is_nothrow_default_constructible_v<_T2>)
      : __base()
  {}

  // copy and move constructors
  _CCCL_HIDE_FROM_ABI pair(const pair&) = default;
  _CCCL_HIDE_FROM_ABI pair(pair&&)      = default;

  // element wise constructors
  template <__select_constructor _Trait = __tuple_select_variadic_copy_constructible_v<__tuple_types<_T1, _T2>>,
            enable_if_t<__can_construct_implicitly<_Trait>, int> = 0>
  _CCCL_API constexpr pair(const _T1& __t1, const _T2& __t2) noexcept(
    is_nothrow_copy_constructible_v<_T1> && is_nothrow_copy_constructible_v<_T2>)
      : __base(__t1, __t2)
  {}
  template <__select_constructor _Trait = __tuple_select_variadic_copy_constructible_v<__tuple_types<_T1, _T2>>,
            enable_if_t<__can_construct_explicitly<_Trait>, int> = 0>
  _CCCL_API explicit constexpr pair(const _T1& __t1, const _T2& __t2) noexcept(
    is_nothrow_copy_constructible_v<_T1> && is_nothrow_copy_constructible_v<_T2>)
      : __base(__t1, __t2)
  {}

  template <class _U1 = _T1,
            class _U2 = _T2,
            __select_constructor _Trait =
              __tuple_select_variadic_constructible_v<__tuple_types<_T1, _T2>, __tuple_types<_U1, _U2>>,
            enable_if_t<__can_construct_implicitly<_Trait>, int> = 0>
  _CCCL_API constexpr pair(_U1&& __u1, _U2&& __u2) noexcept(
    is_nothrow_constructible_v<_T1, _U1> && is_nothrow_constructible_v<_T2, _U2>)
      : __base(::cuda::std::forward<_U1>(__u1), ::cuda::std::forward<_U2>(__u2))
  {}

  template <class _U1 = _T1,
            class _U2 = _T2,
            __select_constructor _Trait =
              __tuple_select_variadic_constructible_v<__tuple_types<_T1, _T2>, __tuple_types<_U1, _U2>>,
            enable_if_t<__can_construct_explicitly<_Trait>, int> = 0>
  _CCCL_API explicit constexpr pair(_U1&& __u1, _U2&& __u2) noexcept(
    is_nothrow_constructible_v<_T1, _U1> && is_nothrow_constructible_v<_T2, _U2>)
      : __base(::cuda::std::forward<_U1>(__u1), ::cuda::std::forward<_U2>(__u2))
  {}

  // converting constructors
  template <class _U1,
            class _U2,
            __select_constructor _Trait = __pair_select_pair_like_constructible_v<pair<_U1, _U2>&, _T1, _T2>,
            enable_if_t<__can_construct_implicitly<_Trait>, int> = 0>
  _CCCL_API constexpr pair(pair<_U1, _U2>& __p) noexcept(
    is_nothrow_constructible_v<_T1, _U1&> && is_nothrow_constructible_v<_T2, _U2&>)
      : __base(__p.first, __p.second)
  {}

  template <class _U1,
            class _U2,
            __select_constructor _Trait = __pair_select_pair_like_constructible_v<pair<_U1, _U2>&, _T1, _T2>,
            enable_if_t<__can_construct_explicitly<_Trait>, int> = 0>
  _CCCL_API explicit constexpr pair(pair<_U1, _U2>& __p) noexcept(
    is_nothrow_constructible_v<_T1, _U1&> && is_nothrow_constructible_v<_T2, _U2&>)
      : __base(__p.first, __p.second)
  {}

  template <class _U1,
            class _U2,
            __select_constructor _Trait = __pair_select_pair_like_constructible_v<const pair<_U1, _U2>&, _T1, _T2>,
            enable_if_t<__can_construct_implicitly<_Trait>, int> = 0>
  _CCCL_API constexpr pair(const pair<_U1, _U2>& __p) noexcept(
    is_nothrow_constructible_v<_T1, const _U1&> && is_nothrow_constructible_v<_T2, const _U2&>)
      : __base(__p.first, __p.second)
  {}

  template <class _U1,
            class _U2,
            __select_constructor _Trait = __pair_select_pair_like_constructible_v<const pair<_U1, _U2>&, _T1, _T2>,
            enable_if_t<__can_construct_explicitly<_Trait>, int> = 0>
  _CCCL_API explicit constexpr pair(const pair<_U1, _U2>& __p) noexcept(
    is_nothrow_constructible_v<_T1, const _U1&> && is_nothrow_constructible_v<_T2, const _U2&>)
      : __base(__p.first, __p.second)
  {}

  template <class _U1,
            class _U2,
            __select_constructor _Trait = __pair_select_pair_like_constructible_v<pair<_U1, _U2>&&, _T1, _T2>,
            enable_if_t<__can_construct_implicitly<_Trait>, int> = 0>
  _CCCL_API constexpr pair(pair<_U1, _U2>&& __p) noexcept(
    is_nothrow_constructible_v<_T1, _U1> && is_nothrow_constructible_v<_T2, _U2>)
      : __base(::cuda::std::move(__p.first), ::cuda::std::move(__p.second))
  {}

  template <class _U1,
            class _U2,
            __select_constructor _Trait = __pair_select_pair_like_constructible_v<pair<_U1, _U2>&&, _T1, _T2>,
            enable_if_t<__can_construct_explicitly<_Trait>, int> = 0>
  _CCCL_API explicit constexpr pair(pair<_U1, _U2>&& __p) noexcept(
    is_nothrow_constructible_v<_T1, _U1> && is_nothrow_constructible_v<_T2, _U2>)
      : __base(::cuda::std::move(__p.first), ::cuda::std::move(__p.second))
  {}

  template <class _U1,
            class _U2,
            __select_constructor _Trait = __pair_select_pair_like_constructible_v<const pair<_U1, _U2>&&, _T1, _T2>,
            enable_if_t<__can_construct_implicitly<_Trait>, int> = 0>
  _CCCL_API constexpr pair(const pair<_U1, _U2>&& __p) noexcept(
    is_nothrow_constructible_v<_T1, const _U1> && is_nothrow_constructible_v<_T2, const _U2>)
      : __base(::cuda::std::move(__p.first), ::cuda::std::move(__p.second))
  {}

  template <class _U1,
            class _U2,
            __select_constructor _Trait = __pair_select_pair_like_constructible_v<const pair<_U1, _U2>&&, _T1, _T2>,
            enable_if_t<__can_construct_explicitly<_Trait>, int> = 0>
  _CCCL_API explicit constexpr pair(const pair<_U1, _U2>&& __p) noexcept(
    is_nothrow_constructible_v<_T1, const _U1> && is_nothrow_constructible_v<_T2, const _U2>)
      : __base(::cuda::std::move(__p.first), ::cuda::std::move(__p.second))
  {}

  // NOLINTBEGIN(bugprone-forwarding-reference-overload)
  _CCCL_EXEC_CHECK_DISABLE
  template <class _UPair,
            enable_if_t<!is_same_v<remove_cvref_t<_UPair>, pair>, int> = 0,
            __select_constructor _Trait = __pair_select_pair_like_constructible_v<_UPair, _T1, _T2>,
            enable_if_t<__can_construct_implicitly<_Trait>, int> = 0>
  _CCCL_API constexpr pair(_UPair&& __p) noexcept(
    is_nothrow_constructible_v<_T1, decltype(::cuda::std::__adl_get<0>(::cuda::std::forward<_UPair>(__p)))>
    && is_nothrow_constructible_v<_T2, decltype(::cuda::std::__adl_get<1>(::cuda::std::forward<_UPair>(__p)))>)
      : __base(
          // __adl_get() specifically will only move the sub-object, it's therefore OK to
          // "move" the outer pair twice
          // NOLINTBEGIN(bugprone-use-after-move)
          ::cuda::std::__adl_get<0>(::cuda::std::forward<_UPair>(__p)),
          ::cuda::std::__adl_get<1>(::cuda::std::forward<_UPair>(__p))
          // NOLINTEND(bugprone-use-after-move)
        )
  {}
  // NOLINTEND(bugprone-forwarding-reference-overload)

  // NOLINTBEGIN(bugprone-forwarding-reference-overload)
  _CCCL_EXEC_CHECK_DISABLE
  template <class _UPair,
            enable_if_t<!is_same_v<remove_cvref_t<_UPair>, pair>, int> = 0,
            __select_constructor _Trait = __pair_select_pair_like_constructible_v<_UPair, _T1, _T2>,
            enable_if_t<__can_construct_explicitly<_Trait>, int> = 0>
  _CCCL_API explicit constexpr pair(_UPair&& __p) noexcept(
    is_nothrow_constructible_v<_T1, decltype(::cuda::std::__adl_get<0>(::cuda::std::forward<_UPair>(__p)))>
    && is_nothrow_constructible_v<_T2, decltype(::cuda::std::__adl_get<1>(::cuda::std::forward<_UPair>(__p)))>)
      : __base(::cuda::std::__adl_get<0>(::cuda::std::forward<_UPair>(__p)),
               ::cuda::std::__adl_get<1>(::cuda::std::forward<_UPair>(__p)))
  {}
  // NOLINTEND(bugprone-forwarding-reference-overload)

  template <class... _Args1, class... _Args2>
  _CCCL_API constexpr pair(piecewise_construct_t __pc,
                           tuple<_Args1...> __first_args,
                           tuple<_Args2...> __second_args) noexcept((is_nothrow_constructible_v<_T1, _Args1...>
                                                                     && is_nothrow_constructible_v<_T2, _Args2...>) )
      : __base(__pc,
               __first_args,
               __second_args,
               __make_tuple_indices_t<sizeof...(_Args1)>(),
               __make_tuple_indices_t<sizeof...(_Args2)>())
  {}

  // assignments
  _CCCL_HIDE_FROM_ABI pair& operator=(const pair&) = default;
  _CCCL_HIDE_FROM_ABI pair& operator=(pair&&)      = default;

  template <class _U1,
            class _U2,
            enable_if_t<is_assignable_v<_T1&, const _U1&>, int> = 0,
            enable_if_t<is_assignable_v<_T2&, const _U2&>, int> = 0>
  _CCCL_API constexpr pair& operator=(const pair<_U1, _U2>& __p) noexcept(
    is_nothrow_assignable_v<_T1&, const _U1&> && is_nothrow_assignable_v<_T2&, const _U2&>)
  {
    this->first  = __p.first;
    this->second = __p.second;
    return *this;
  }

  template <class _U1,
            class _U2,
            enable_if_t<is_assignable_v<const _T1&, const _U1&>, int> = 0,
            enable_if_t<is_assignable_v<const _T2&, const _U2&>, int> = 0>
  _CCCL_API constexpr const pair& operator=(const pair<_U1, _U2>& __p) const
    noexcept(is_nothrow_assignable_v<const _T1&, const _U1&> && is_nothrow_assignable_v<const _T2&, const _U2&>)
  {
    this->first  = __p.first;
    this->second = __p.second;
    return *this;
  }

  template <class _U1,
            class _U2,
            enable_if_t<is_assignable_v<_T1&, _U1>, int> = 0,
            enable_if_t<is_assignable_v<_T2&, _U2>, int> = 0>
  _CCCL_API constexpr pair&
  operator=(pair<_U1, _U2>&& __p) noexcept(is_nothrow_assignable_v<_T1&, _U1> && is_nothrow_assignable_v<_T2&, _U2>)
  {
    this->first  = ::cuda::std::forward<_U1>(__p.first);
    this->second = ::cuda::std::forward<_U2>(__p.second);
    return *this;
  }

  template <class _U1,
            class _U2,
            enable_if_t<is_assignable_v<const _T1&, _U1>, int> = 0,
            enable_if_t<is_assignable_v<const _T2&, _U2>, int> = 0>
  _CCCL_API constexpr const pair& operator=(pair<_U1, _U2>&& __p) const
    noexcept(is_nothrow_assignable_v<const _T1&, _U1> && is_nothrow_assignable_v<const _T2&, _U2>)
  {
    this->first  = ::cuda::std::forward<_U1>(__p.first);
    this->second = ::cuda::std::forward<_U2>(__p.second);
    return *this;
  }

  // ``get`` will specifically only move the sub-object, it's therefore OK to
  // "move" the outer pair twice
  // NOLINTBEGIN(bugprone-use-after-move)
  _CCCL_EXEC_CHECK_DISABLE
  template <class _UPair,
            __select_assignment _Trait = __pair_select_pair_like_assignable_v</*__is_const=*/false, _UPair, _T1, _T2>,
            enable_if_t<__can_assign<_Trait>, int> = 0>
  _CCCL_API constexpr pair& operator=(_UPair&& __p) noexcept(__can_nothrow_assign<_Trait>)
  {
    using ::cuda::std::get;
    this->first  = get<0>(::cuda::std::forward<_UPair>(__p));
    this->second = get<1>(::cuda::std::forward<_UPair>(__p));
    return *this;
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <class _UPair,
            __select_assignment _Trait = __pair_select_pair_like_assignable_v</*__is_const=*/true, _UPair, _T1, _T2>,
            enable_if_t<__can_assign<_Trait>, int> = 0>
  _CCCL_API constexpr const pair& operator=(_UPair&& __p) const noexcept(__can_nothrow_assign<_Trait>)
  {
    using ::cuda::std::get;
    this->first  = get<0>(::cuda::std::forward<_UPair>(__p));
    this->second = get<1>(::cuda::std::forward<_UPair>(__p));
    return *this;
  }
  // NOLINTEND(bugprone-use-after-move)

  _CCCL_API constexpr void swap(pair& __p) noexcept(is_nothrow_swappable_v<_T1> && is_nothrow_swappable_v<_T2>)
  {
    using ::cuda::std::swap;
    swap(this->first, __p.first);
    swap(this->second, __p.second);
  }

#if _CCCL_STD_VER >= 2023
  _CCCL_API constexpr void swap(const pair& __p) const
    noexcept(is_nothrow_swappable_v<const _T1> && is_nothrow_swappable_v<const _T2>)
  {
    using ::cuda::std::swap;
    swap(this->first, __p.first);
    swap(this->second, __p.second);
  }
#endif // _CCCL_STD_VER >= 2023

#if _CCCL_HOSTED()
  _CCCL_HOST_API constexpr operator ::std::pair<_T1, _T2>() const
  {
    return {this->first, this->second};
  }
#endif // _CCCL_HOSTED()
};

template <class _T1, class _T2>
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES pair(_T1, _T2) -> pair<_T1, _T2>;

// [pairs.spec], specialized algorithms

_CCCL_EXEC_CHECK_DISABLE
template <class _T1, class _T2>
_CCCL_API constexpr bool operator==(const pair<_T1, _T2>& __x, const pair<_T1, _T2>& __y)
{
  return __x.first == __y.first && __x.second == __y.second;
}

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

_CCCL_EXEC_CHECK_DISABLE
template <class _T1, class _T2>
_CCCL_API constexpr common_comparison_category_t<__synth_three_way_result<_T1>, __synth_three_way_result<_T2>>
operator<=>(const pair<_T1, _T2>& __x, const pair<_T1, _T2>& __y)
{
  if (auto __c = ::cuda::std::__synth_three_way(__x.first, __y.first); __c != 0)
  {
    return __c;
  }
  return ::cuda::std::__synth_three_way(__x.second, __y.second);
}

#else // ^^^ _LIBCUDACXX_HAS_SPACESHIP_OPERATOR() ^^^ / vvv !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR() vvv

_CCCL_EXEC_CHECK_DISABLE
template <class _T1, class _T2>
_CCCL_API constexpr bool operator!=(const pair<_T1, _T2>& __x, const pair<_T1, _T2>& __y)
{
  return !(__x == __y);
}

_CCCL_EXEC_CHECK_DISABLE
template <class _T1, class _T2>
_CCCL_API constexpr bool operator<(const pair<_T1, _T2>& __x, const pair<_T1, _T2>& __y)
{
  return __x.first < __y.first || (!(__y.first < __x.first) && __x.second < __y.second);
}

_CCCL_EXEC_CHECK_DISABLE
template <class _T1, class _T2>
_CCCL_API constexpr bool operator>(const pair<_T1, _T2>& __x, const pair<_T1, _T2>& __y)
{
  return __y < __x;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _T1, class _T2>
_CCCL_API constexpr bool operator>=(const pair<_T1, _T2>& __x, const pair<_T1, _T2>& __y)
{
  return !(__x < __y);
}

_CCCL_EXEC_CHECK_DISABLE
template <class _T1, class _T2>
_CCCL_API constexpr bool operator<=(const pair<_T1, _T2>& __x, const pair<_T1, _T2>& __y)
{
  return !(__y < __x);
}

#endif // !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

#if _CCCL_STD_VER >= 2023
template <class _T1, class _T2, class _U1, class _U2, template <class> class _TQual, template <class> class _UQual>
  requires requires {
    typename pair<common_reference_t<_TQual<_T1>, _UQual<_U1>>, common_reference_t<_TQual<_T2>, _UQual<_U2>>>;
  }
struct basic_common_reference<pair<_T1, _T2>, pair<_U1, _U2>, _TQual, _UQual>
{
  using type = pair<common_reference_t<_TQual<_T1>, _UQual<_U1>>, common_reference_t<_TQual<_T2>, _UQual<_U2>>>;
};

template <class _T1, class _T2, class _U1, class _U2>
  requires requires { typename pair<common_type_t<_T1, _U1>, common_type_t<_T2, _U2>>; }
struct common_type<pair<_T1, _T2>, pair<_U1, _U2>>
{
  using type = pair<common_type_t<_T1, _U1>, common_type_t<_T2, _U2>>;
};
#endif // _CCCL_STD_VER >= 2023

_CCCL_TEMPLATE(class _T1, class _T2)
_CCCL_REQUIRES(is_swappable_v<_T1> _CCCL_AND is_swappable_v<_T2>)
_CCCL_API constexpr void
swap(pair<_T1, _T2>& __x, pair<_T1, _T2>& __y) noexcept((is_nothrow_swappable_v<_T1> && is_nothrow_swappable_v<_T2>) )
{
  __x.swap(__y);
}

_CCCL_TEMPLATE(class _T1, class _T2)
_CCCL_REQUIRES(is_swappable_v<const _T1> _CCCL_AND is_swappable_v<const _T2>)
_CCCL_API constexpr void swap(const pair<_T1, _T2>& __x, const pair<_T1, _T2>& __y) noexcept(
  (is_nothrow_swappable_v<const _T1> && is_nothrow_swappable_v<const _T2>) )
{
  __x.swap(__y);
}

template <class _T1, class _T2>
_CCCL_API constexpr pair<unwrap_ref_decay_t<_T1>, unwrap_ref_decay_t<_T2>> make_pair(_T1&& __t1, _T2&& __t2)
{
  return pair<unwrap_ref_decay_t<_T1>, unwrap_ref_decay_t<_T2>>(
    ::cuda::std::forward<_T1>(__t1), ::cuda::std::forward<_T2>(__t2));
}

template <size_t _Ip>
struct __get_pair;

template <>
struct __get_pair<0>
{
  template <class _T1, class _T2>
  [[nodiscard]] _CCCL_API static constexpr _T1& get(pair<_T1, _T2>& __p) noexcept
  {
    return __p.first;
  }

  template <class _T1, class _T2>
  [[nodiscard]] _CCCL_API static constexpr const _T1& get(const pair<_T1, _T2>& __p) noexcept
  {
    return __p.first;
  }

  template <class _T1, class _T2>
  [[nodiscard]] _CCCL_API static constexpr _T1&& get(pair<_T1, _T2>&& __p) noexcept
  {
    return ::cuda::std::forward<_T1>(__p.first);
  }

  template <class _T1, class _T2>
  [[nodiscard]] _CCCL_API static constexpr const _T1&& get(const pair<_T1, _T2>&& __p) noexcept
  {
    return ::cuda::std::forward<const _T1>(__p.first);
  }
};

template <>
struct __get_pair<1>
{
  template <class _T1, class _T2>
  [[nodiscard]] _CCCL_API static constexpr _T2& get(pair<_T1, _T2>& __p) noexcept
  {
    return __p.second;
  }

  template <class _T1, class _T2>
  [[nodiscard]] _CCCL_API static constexpr const _T2& get(const pair<_T1, _T2>& __p) noexcept
  {
    return __p.second;
  }

  template <class _T1, class _T2>
  [[nodiscard]] _CCCL_API static constexpr _T2&& get(pair<_T1, _T2>&& __p) noexcept
  {
    return ::cuda::std::forward<_T2>(__p.second);
  }

  template <class _T1, class _T2>
  [[nodiscard]] _CCCL_API static constexpr const _T2&& get(const pair<_T1, _T2>&& __p) noexcept
  {
    return ::cuda::std::forward<const _T2>(__p.second);
  }
};

template <size_t _Ip, class _T1, class _T2>
[[nodiscard]] _CCCL_API constexpr tuple_element_t<_Ip, pair<_T1, _T2>>& get(pair<_T1, _T2>& __p) noexcept
{
  return ::cuda::std::__get_pair<_Ip>::get(__p);
}

template <size_t _Ip, class _T1, class _T2>
[[nodiscard]] _CCCL_API constexpr const tuple_element_t<_Ip, pair<_T1, _T2>>& get(const pair<_T1, _T2>& __p) noexcept
{
  return ::cuda::std::__get_pair<_Ip>::get(__p);
}

template <size_t _Ip, class _T1, class _T2>
[[nodiscard]] _CCCL_API constexpr tuple_element_t<_Ip, pair<_T1, _T2>>&& get(pair<_T1, _T2>&& __p) noexcept
{
  return ::cuda::std::__get_pair<_Ip>::get(::cuda::std::move(__p));
}

template <size_t _Ip, class _T1, class _T2>
[[nodiscard]] _CCCL_API constexpr const tuple_element_t<_Ip, pair<_T1, _T2>>&& get(const pair<_T1, _T2>&& __p) noexcept
{
  return ::cuda::std::__get_pair<_Ip>::get(::cuda::std::move(__p));
}

template <class _T1, class _T2>
[[nodiscard]] _CCCL_API constexpr _T1& get(pair<_T1, _T2>& __p) noexcept
{
  return ::cuda::std::__get_pair<0>::get(__p);
}

template <class _T1, class _T2>
[[nodiscard]] _CCCL_API constexpr const _T1& get(const pair<_T1, _T2>& __p) noexcept
{
  return ::cuda::std::__get_pair<0>::get(__p);
}

template <class _T1, class _T2>
[[nodiscard]] _CCCL_API constexpr _T1&& get(pair<_T1, _T2>&& __p) noexcept
{
  return ::cuda::std::__get_pair<0>::get(::cuda::std::move(__p));
}

template <class _T1, class _T2>
[[nodiscard]] _CCCL_API constexpr const _T1&& get(const pair<_T1, _T2>&& __p) noexcept
{
  return ::cuda::std::__get_pair<0>::get(::cuda::std::move(__p));
}

template <class _T1, class _T2>
[[nodiscard]] _CCCL_API constexpr _T1& get(pair<_T2, _T1>& __p) noexcept
{
  return ::cuda::std::__get_pair<1>::get(__p);
}

template <class _T1, class _T2>
[[nodiscard]] _CCCL_API constexpr const _T1& get(const pair<_T2, _T1>& __p) noexcept
{
  return ::cuda::std::__get_pair<1>::get(__p);
}

template <class _T1, class _T2>
[[nodiscard]] _CCCL_API constexpr _T1&& get(pair<_T2, _T1>&& __p) noexcept
{
  return ::cuda::std::__get_pair<1>::get(::cuda::std::move(__p));
}

template <class _T1, class _T2>
[[nodiscard]] _CCCL_API constexpr const _T1&& get(const pair<_T2, _T1>&& __p) noexcept
{
  return ::cuda::std::__get_pair<1>::get(::cuda::std::move(__p));
}

// specialize cuda::std::tuple_size and cuda::std::tuple_element for std::pair and cuda::std::pair

#if _CCCL_HAS_HOST_STD_LIB()
template <class _Tp, class _Up>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_size<::std::pair<_Tp, _Up>> : integral_constant<size_t, 2>
{};

template <size_t _Ip, class _Tp, class _Up>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_element<_Ip, ::std::pair<_Tp, _Up>>
{
  static_assert(_Ip < 2, "Index out of bounds in cuda::std::tuple_element<std::pair<_Tp, _Up>>");
};
template <class _Tp, class _Up>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_element<0, ::std::pair<_Tp, _Up>>
{
  using type _CCCL_NODEBUG_ALIAS = _Tp;
};
template <class _Tp, class _Up>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_element<1, ::std::pair<_Tp, _Up>>
{
  using type _CCCL_NODEBUG_ALIAS = _Up;
};
#endif // _CCCL_HAS_HOST_STD_LIB()

template <class _Tp, class _Up>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_size<pair<_Tp, _Up>> : integral_constant<size_t, 2>
{};

template <size_t _Ip, class _Tp, class _Up>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_element<_Ip, pair<_Tp, _Up>>
{
  static_assert(_Ip < 2, "Index out of bounds in cuda::std::tuple_element<cuda::std::pair<_Tp, _Up>>");
};
template <class _Tp, class _Up>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_element<0, pair<_Tp, _Up>>
{
  using type _CCCL_NODEBUG_ALIAS = _Tp;
};
template <class _Tp, class _Up>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_element<1, pair<_Tp, _Up>>
{
  using type _CCCL_NODEBUG_ALIAS = _Up;
};

_CCCL_END_NAMESPACE_CUDA_STD

// tuple protocol for cuda::std::pair

_CCCL_BEGIN_NAMESPACE_STD

template <class _Tp, class _Up>
struct tuple_size<::cuda::std::pair<_Tp, _Up>> : ::cuda::std::integral_constant<::cuda::std::size_t, 2>
{};

template <::cuda::std::size_t _Ip, class _Tp, class _Up>
struct tuple_element<_Ip, ::cuda::std::pair<_Tp, _Up>>
{
  static_assert(_Ip < 2, "Index out of bounds in std::tuple_element<cuda::std::pair<_Tp, _Up>>");
};
template <class _Tp, class _Up>
struct tuple_element<0, ::cuda::std::pair<_Tp, _Up>>
{
  using type _CCCL_NODEBUG_ALIAS = _Tp;
};
template <class _Tp, class _Up>
struct tuple_element<1, ::cuda::std::pair<_Tp, _Up>>
{
  using type _CCCL_NODEBUG_ALIAS = _Up;
};

_CCCL_END_NAMESPACE_STD

_CCCL_END_NV_DIAG_SUPPRESS()

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___UTILITY_PAIR_H
