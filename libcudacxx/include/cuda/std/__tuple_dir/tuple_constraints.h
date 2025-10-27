//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TUPLE_TUPLE_CONSTRAINTS_H
#define _CUDA_STD___TUPLE_TUPLE_CONSTRAINTS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__fwd/tuple.h>
#include <cuda/std/__tuple_dir/make_tuple_types.h>
#include <cuda/std/__tuple_dir/sfinae_helpers.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/disjunction.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_copy_assignable.h>
#include <cuda/std/__type_traits/is_default_constructible.h>
#include <cuda/std/__type_traits/is_implicitly_default_constructible.h>
#include <cuda/std/__type_traits/is_move_assignable.h>
#include <cuda/std/__type_traits/is_nothrow_assignable.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_copy_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_default_constructible.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/lazy.h>
#include <cuda/std/__type_traits/remove_cvref.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class>
inline constexpr bool __tuple_all_default_constructible_v = false;

template <class... _Tp>
inline constexpr bool __tuple_all_default_constructible_v<__tuple_types<_Tp...>> =
  (is_default_constructible_v<_Tp> && ...);

template <class... _Tp>
inline constexpr bool __tuple_all_copy_assignable_v = (is_copy_assignable_v<_Tp> && ...);

template <class... _Tp>
inline constexpr bool __tuple_all_move_assignable_v = (is_move_assignable_v<_Tp> && ...);

struct __invalid_tuple_constraints
{
  static constexpr bool __implicit_constructible = false;
  static constexpr bool __explicit_constructible = false;
  static constexpr bool __nothrow_constructible  = false;
};

template <class... _Tp>
struct __tuple_constraints
{
  static constexpr bool __default_constructible = (is_default_constructible_v<_Tp> && ...);

  static constexpr bool __nothrow_default_constructible = (is_nothrow_default_constructible_v<_Tp> && ...);

  static constexpr bool __implicit_default_constructible = (__is_implicitly_default_constructible<_Tp>::value && ...);

  static constexpr bool __explicit_default_constructible = __default_constructible && !__implicit_default_constructible;

  static constexpr bool __implicit_variadic_copy_constructible =
    __tuple_constructible<tuple<const _Tp&...>, tuple<_Tp...>>::value
    && __tuple_convertible<tuple<const _Tp&...>, tuple<_Tp...>>::value;

  static constexpr bool __explicit_variadic_copy_constructible =
    __tuple_constructible<tuple<const _Tp&...>, tuple<_Tp...>>::value
    && !__tuple_convertible<tuple<const _Tp&...>, tuple<_Tp...>>::value;

  static constexpr bool __nothrow_variadic_copy_constructible = (is_nothrow_copy_constructible_v<_Tp> && ...);

  template <class... _Args>
  struct _PackExpandsToThisTuple : false_type
  {};

  template <class _Arg>
  struct _PackExpandsToThisTuple<_Arg>
  {
    static constexpr bool value = is_same_v<remove_cvref_t<_Arg>, tuple<_Tp...>>;
  };

  template <class... _Args>
  struct __variadic_constraints
  {
    static constexpr bool __implicit_constructible =
      __tuple_constructible<tuple<_Args...>, tuple<_Tp...>>::value
      && __tuple_convertible<tuple<_Args...>, tuple<_Tp...>>::value;

    static constexpr bool __explicit_constructible =
      __tuple_constructible<tuple<_Args...>, tuple<_Tp...>>::value
      && !__tuple_convertible<tuple<_Args...>, tuple<_Tp...>>::value;

    static constexpr bool __nothrow_constructible = (is_nothrow_constructible_v<_Tp, _Args> && ...);
  };

  template <class... _Args>
  struct __variadic_constraints_less_rank
  {
    static constexpr bool __implicit_constructible =
      __tuple_constructible<tuple<_Args...>, __make_tuple_types_t<tuple<_Tp...>, sizeof...(_Args)>>::value
      && __tuple_convertible<tuple<_Args...>, __make_tuple_types_t<tuple<_Tp...>, sizeof...(_Args)>>::value
      && __tuple_all_default_constructible_v<__make_tuple_types_t<tuple<_Tp...>, sizeof...(_Tp), sizeof...(_Args)>>;

    static constexpr bool __explicit_constructible =
      __tuple_constructible<tuple<_Args...>, __make_tuple_types_t<tuple<_Tp...>, sizeof...(_Args)>>::value
      && !__tuple_convertible<tuple<_Args...>, __make_tuple_types_t<tuple<_Tp...>, sizeof...(_Args)>>::value
      && __tuple_all_default_constructible_v<__make_tuple_types_t<tuple<_Tp...>, sizeof...(_Tp), sizeof...(_Args)>>;
  };

  template <class _Tuple>
  struct __valid_tuple_like_constraints
  {
    static constexpr bool __implicit_constructible =
      __tuple_constructible<_Tuple, tuple<_Tp...>>::value && __tuple_convertible<_Tuple, tuple<_Tp...>>::value;

    static constexpr bool __explicit_constructible =
      __tuple_constructible<_Tuple, tuple<_Tp...>>::value && !__tuple_convertible<_Tuple, tuple<_Tp...>>::value;
  };

  template <class _Tuple>
  struct __valid_tuple_like_constraints_rank_one
  {
    template <class _Tuple2>
    struct _PreferTupleLikeConstructorImpl
        : _Or<
            // Don't attempt the two checks below if the tuple we are given
            // has the same type as this tuple.
            _IsSame<remove_cvref_t<_Tuple2>, tuple<_Tp...>>,
            _Lazy<_And, _Not<is_constructible<_Tp..., _Tuple2>>, _Not<is_convertible<_Tuple2, _Tp...>>>>
    {};

    // This trait is used to disable the tuple-like constructor when
    // the UTypes... constructor should be selected instead.
    // See LWG issue #2549.
    template <class _Tuple2>
    using _PreferTupleLikeConstructor = _PreferTupleLikeConstructorImpl<_Tuple2>;

    static constexpr bool __implicit_constructible =
      __tuple_constructible<_Tuple, tuple<_Tp...>>::value && __tuple_convertible<_Tuple, tuple<_Tp...>>::value
      && _PreferTupleLikeConstructor<_Tuple>::value;

    static constexpr bool __explicit_constructible =
      __tuple_constructible<_Tuple, tuple<_Tp...>>::value && !__tuple_convertible<_Tuple, tuple<_Tp...>>::value
      && _PreferTupleLikeConstructor<_Tuple>::value;
  };

  template <class _Tuple>
  using __tuple_like_constraints =
    conditional_t<sizeof...(_Tp) == 1,
                  __valid_tuple_like_constraints_rank_one<_Tuple>,
                  __valid_tuple_like_constraints<_Tuple>>;
};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TUPLE_TUPLE_CONSTRAINTS_H
