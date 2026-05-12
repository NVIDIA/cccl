//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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
#include <cuda/std/__tuple_dir/tuple_element.h>
#include <cuda/std/__tuple_dir/tuple_like_ext.h>
#include <cuda/std/__tuple_dir/tuple_size.h>
#include <cuda/std/__tuple_dir/tuple_types.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/conjunction.h>
#include <cuda/std/__type_traits/disjunction.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_assignable.h>
#include <cuda/std/__type_traits/is_comparable.h>
#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__type_traits/is_convertible.h>
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
#include <cuda/std/__type_traits/remove_reference.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

struct allocator_arg_t;

template <class... _Tp>
inline constexpr bool __tuple_all_copy_assignable_v = (is_copy_assignable_v<_Tp> && ...);

template <class... _Tp>
inline constexpr bool __tuple_all_move_assignable_v = (is_move_assignable_v<_Tp> && ...);

// Traits forwarding to `__tuple_types`
template <class>
inline constexpr bool __tuple_types_all_default_constructible_v = false;

template <class... _Tp>
inline constexpr bool __tuple_types_all_default_constructible_v<__tuple_types<_Tp...>> =
  (is_default_constructible_v<_Tp> && ...);

template <class, class>
inline constexpr bool __tuple_types_same_size = false;

template <class... _Tp, class... _Up>
inline constexpr bool __tuple_types_same_size<__tuple_types<_Tp...>, __tuple_types<_Up...>> =
  sizeof...(_Tp) == sizeof...(_Up);

// __tuple_constructible
template <class _From, class _To, bool = __tuple_types_same_size<_From, _To>>
inline constexpr bool __tuple_types_constructible = false;

template <class... _From, class... _To>
inline constexpr bool __tuple_types_constructible<__tuple_types<_From...>, __tuple_types<_To...>, true> =
  (is_constructible_v<_To, _From> && ...);

template <class _From, class _To, bool = __tuple_like_ext<remove_reference_t<_From>>, bool = __tuple_like_ext<_To>>
inline constexpr bool __tuple_constructible = false;

template <class _From, class _To>
inline constexpr bool __tuple_constructible<_From, _To, true, true> =
  __tuple_types_constructible<__make_tuple_types_t<_From>, __make_tuple_types_t<_To>>;

template <class _Tp, class _Up>
struct __tuple_constructible_struct
{
  static constexpr bool value = __tuple_constructible<_Tp, _Up>;
};

// __tuple_nothrow_constructible
template <class _From, class _To, bool = __tuple_types_constructible<_From, _To>>
inline constexpr bool __tuple_types_nothrow_constructible = false;

template <class... _From, class... _To>
inline constexpr bool __tuple_types_nothrow_constructible<__tuple_types<_From...>, __tuple_types<_To...>, true> =
  (is_nothrow_constructible_v<_To, _From> && ...);

template <class _From, class _To, bool = __tuple_constructible<_From, _To>>
inline constexpr bool __tuple_nothrow_constructible = false;

template <class _From, class _To>
inline constexpr bool __tuple_nothrow_constructible<_From, _To, true> =
  __tuple_types_nothrow_constructible<__make_tuple_types_t<_From>, __make_tuple_types_t<_To>>;

// __tuple_convertible
template <class _From, class _To, bool = __tuple_types_same_size<_From, _To>>
inline constexpr bool __tuple_types_convertible = false;

template <class... _From, class... _To>
inline constexpr bool __tuple_types_convertible<__tuple_types<_From...>, __tuple_types<_To...>, true> =
  (is_convertible_v<_From, _To> && ...);

template <class _From, class _To, bool = __tuple_like_ext<remove_reference_t<_From>>, bool = __tuple_like_ext<_To>>
inline constexpr bool __tuple_convertible = false;

template <class _From, class _To>
inline constexpr bool __tuple_convertible<_From, _To, true, true> =
  __tuple_types_convertible<__make_tuple_types_t<_From>, __make_tuple_types_t<_To>>;

// __tuple_assignable
template <class _From, class _To, bool = __tuple_types_same_size<_From, _To>>
inline constexpr bool __tuple_types_assignable = false;

template <class... _From, class... _To>
inline constexpr bool __tuple_types_assignable<__tuple_types<_From...>, __tuple_types<_To...>, true> =
  (is_assignable_v<_To, _From> && ...);

template <class _From, class _To, bool = __tuple_like_ext<remove_reference_t<_From>>, bool = __tuple_like_ext<_To>>
inline constexpr bool __tuple_assignable = false;

template <class _From, class _To>
inline constexpr bool __tuple_assignable<_From, _To, true, true> =
  __tuple_types_assignable<__make_tuple_types_t<_From>, __make_tuple_types_t<_To&>>;

// __tuple_nothrow_assignable
template <class _From, class _To, bool = __tuple_types_same_size<_From, _To>>
inline constexpr bool __tuple_types_nothrow_assignable = false;

template <class... _From, class... _To>
inline constexpr bool __tuple_types_nothrow_assignable<__tuple_types<_From...>, __tuple_types<_To...>, true> =
  (is_nothrow_assignable_v<_To, _From> && ...);

template <class _From, class _To, bool = __tuple_assignable<_From, _To>>
inline constexpr bool __tuple_nothrow_assignable = false;

template <class _From, class _To>
inline constexpr bool __tuple_nothrow_assignable<_From, _To, true> =
  __tuple_types_assignable<__make_tuple_types_t<_From>, __make_tuple_types_t<_To&>>;

// __tuple_like_with_size
template <class _Tuple, size_t _ExpectedSize, bool = __tuple_like_ext<remove_cvref_t<_Tuple>>>
inline constexpr bool __tuple_like_with_size = false;

template <class _Tuple, size_t _ExpectedSize>
inline constexpr bool __tuple_like_with_size<_Tuple, _ExpectedSize, true> =
  _ExpectedSize == tuple_size<remove_cvref_t<_Tuple>>::value;

struct __invalid_tuple_constraints
{
  static constexpr bool __implicit_constructible = false;
  static constexpr bool __explicit_constructible = false;
  static constexpr bool __nothrow_constructible  = false;

  static constexpr bool __equality_comparable          = false;
  static constexpr bool __nothrow_equality_comparable  = false;
  static constexpr bool __less_than_comparable         = false;
  static constexpr bool __nothrow_less_than_comparable = false;
};

template <class... _Tp>
struct __tuple_constraints
{
  static constexpr bool __default_constructible = (is_default_constructible_v<_Tp> && ...);

  static constexpr bool __nothrow_default_constructible = (is_nothrow_default_constructible_v<_Tp> && ...);

  static constexpr bool __implicit_default_constructible = (__is_implicitly_default_constructible<_Tp>::value && ...);

  static constexpr bool __explicit_default_constructible = __default_constructible && !__implicit_default_constructible;

  static constexpr bool __implicit_variadic_copy_constructible =
    __tuple_constructible<__tuple_types<const _Tp&...>, __tuple_types<_Tp...>>
    && __tuple_convertible<__tuple_types<const _Tp&...>, __tuple_types<_Tp...>>;

  static constexpr bool __explicit_variadic_copy_constructible =
    __tuple_constructible<__tuple_types<const _Tp&...>, __tuple_types<_Tp...>>
    && !__tuple_convertible<__tuple_types<const _Tp&...>, __tuple_types<_Tp...>>;

  static constexpr bool __nothrow_variadic_copy_constructible = (is_nothrow_copy_constructible_v<_Tp> && ...);

  template <class... _Args>
  struct __variadic_constraints
  {
    // 12.3: otherwise, true_type
    template <class... _Up>
    static constexpr bool __disambiguation_constraints = true;

    // 12.1: negation<is_same<remove_cvref_t<U0>, tuple>> if sizeof...(Types) is 1
    template <class _U0>
    static constexpr bool __disambiguation_constraints<_U0> = !is_same_v<remove_cvref_t<_U0>, tuple<_Tp...>>;

    // 12.2: otherwise, bool_constant<!is_same_v<remove_cvref_t<U0>, allocator_arg_t> ||
    //       is_same_v<remove_cvref_t<T0>, allocator_arg_t>> if sizeof...(Types) is 2 or 3
    template <class _U0, class _U1>
    static constexpr bool __disambiguation_constraints<_U0, _U1> =
      !is_same_v<remove_cvref_t<_U0>, allocator_arg_t>
      || is_same_v<remove_cvref_t<__type_index_c<0, _Tp...>>, allocator_arg_t>;

    template <class _U0, class _U1, class _U2>
    static constexpr bool __disambiguation_constraints<_U0, _U1, _U2> = __disambiguation_constraints<_U0, _U1>;

    // Must be a function since the is_constructible check needs to be lazy
    [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL bool __check_constructible() noexcept
    {
      if constexpr (__disambiguation_constraints<_Args...>)
      {
        return _And<is_constructible<_Tp, _Args>...>::value;
      }
      else
      {
        return false;
      }
    }

    static constexpr bool __constructible = __check_constructible();

    static constexpr bool __implicit_constructible =
      __constructible && __tuple_convertible<__tuple_types<_Args...>, __tuple_types<_Tp...>>;
    static constexpr bool __explicit_constructible =
      __constructible && !__tuple_convertible<__tuple_types<_Args...>, __tuple_types<_Tp...>>;
    static constexpr bool __nothrow_constructible = (is_nothrow_constructible_v<_Tp, _Args> && ...);
  };

  template <class... _Args>
  struct __variadic_constraints_less_rank
  {
    static constexpr bool __implicit_constructible =
      __tuple_constructible<__tuple_types<_Args...>, __make_tuple_types_t<__tuple_types<_Tp...>, sizeof...(_Args)>>
      && __tuple_convertible<__tuple_types<_Args...>, __make_tuple_types_t<__tuple_types<_Tp...>, sizeof...(_Args)>>
      && __tuple_types_all_default_constructible_v<
        __make_tuple_types_t<__tuple_types<_Tp...>, sizeof...(_Tp), sizeof...(_Args)>>;

    static constexpr bool __explicit_constructible =
      __tuple_constructible<__tuple_types<_Args...>, __make_tuple_types_t<__tuple_types<_Tp...>, sizeof...(_Args)>>
      && !__tuple_convertible<__tuple_types<_Args...>, __make_tuple_types_t<__tuple_types<_Tp...>, sizeof...(_Args)>>
      && __tuple_types_all_default_constructible_v<
        __make_tuple_types_t<__tuple_types<_Tp...>, sizeof...(_Tp), sizeof...(_Args)>>;
  };

  template <class _Tuple>
  struct __valid_tuple_like_constraints
  {
    static constexpr bool __implicit_constructible =
      __tuple_constructible<_Tuple, __tuple_types<_Tp...>> && __tuple_convertible<_Tuple, __tuple_types<_Tp...>>;

    static constexpr bool __explicit_constructible =
      __tuple_constructible<_Tuple, __tuple_types<_Tp...>> && !__tuple_convertible<_Tuple, __tuple_types<_Tp...>>;
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
      __tuple_constructible<_Tuple, __tuple_types<_Tp...>> && __tuple_convertible<_Tuple, __tuple_types<_Tp...>>
      && _PreferTupleLikeConstructor<_Tuple>::value;

    static constexpr bool __explicit_constructible =
      __tuple_constructible<_Tuple, __tuple_types<_Tp...>> && !__tuple_convertible<_Tuple, __tuple_types<_Tp...>>
      && _PreferTupleLikeConstructor<_Tuple>::value;
  };

  template <class _Tuple>
  using __tuple_like_constraints =
    conditional_t<sizeof...(_Tp) == 1,
                  __valid_tuple_like_constraints_rank_one<_Tuple>,
                  __valid_tuple_like_constraints<_Tuple>>;

  template <class _Tuple, class _DecayedOtherTuple = remove_cvref_t<_Tuple>>
  struct __valid_utypes_constraints_rank_one
  {
    static constexpr bool __implicit_constructible = false;
    static constexpr bool __explicit_constructible = false;
  };

  template <class _Tuple, class... _Up>
  struct __valid_utypes_constraints_rank_one<_Tuple, tuple<_Up...>>
  {
    // Use consteval function to lazily evaluate the conditions.
    [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL bool __check_enable_singleton_utype_ctor() noexcept
    {
      if constexpr ((is_same_v<_Tp, _Up> && ...))
      { // NOLINT(bugprone-branch-clone)
        return false;
      }
      else if constexpr ((is_convertible_v<_Tuple, _Tp> && ...))
      {
        return false;
      }
      else
      {
        return !(is_constructible_v<_Tp, _Tuple> && ...);
      }
    }

    static constexpr bool __enable_singleton_utype_ctor = __check_enable_singleton_utype_ctor();

    static constexpr bool __implicit_constructible =
      __enable_singleton_utype_ctor && __valid_tuple_like_constraints_rank_one<_Tuple>::__implicit_constructible;

    static constexpr bool __explicit_constructible =
      __enable_singleton_utype_ctor && __valid_tuple_like_constraints_rank_one<_Tuple>::__explicit_constructible;
  };

  template <class _Tuple>
  using __utypes_tuple_constraints =
    conditional_t<sizeof...(_Tp) == 1,
                  __valid_utypes_constraints_rank_one<_Tuple>,
                  __valid_tuple_like_constraints<_Tuple>>;

  template <class... _Up>
  struct __comparison
  {
    static constexpr bool __equality_comparable         = (__is_cpp17_equality_comparable_v<_Tp, _Up> && ...);
    static constexpr bool __nothrow_equality_comparable = (__is_cpp17_nothrow_equality_comparable_v<_Tp, _Up> && ...);

    static constexpr bool __less_than_comparable         = (__is_cpp17_less_than_comparable_v<_Tp, _Up> && ...);
    static constexpr bool __nothrow_less_than_comparable = (__is_cpp17_nothrow_less_than_comparable_v<_Tp, _Up> && ...);
  };
};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TUPLE_TUPLE_CONSTRAINTS_H
