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
#include <cuda/std/__memory/allocator_arg_t.h>
#include <cuda/std/__tuple_dir/make_tuple_types.h>
#include <cuda/std/__tuple_dir/tuple_element.h>
#include <cuda/std/__tuple_dir/tuple_like_ext.h>
#include <cuda/std/__tuple_dir/tuple_size.h>
#include <cuda/std/__tuple_dir/tuple_types.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/disjunction.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_assignable.h>
#include <cuda/std/__type_traits/is_comparable.h>
#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_copy_assignable.h>
#include <cuda/std/__type_traits/is_copy_constructible.h>
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

template <class... _Types>
inline constexpr bool __tuple_all_copy_assignable_v = (is_copy_assignable_v<_Types> && ...);

template <class... _Types>
inline constexpr bool __tuple_all_move_assignable_v = (is_move_assignable_v<_Types> && ...);

// Traits forwarding to `__tuple_types`
template <class>
inline constexpr bool __tuple_types_all_default_constructible_v = false;

template <class... _Types>
inline constexpr bool __tuple_types_all_default_constructible_v<__tuple_types<_Types...>> =
  (is_default_constructible_v<_Types> && ...);

template <class, class>
inline constexpr bool __tuple_types_same_size = false;

template <class... _Types, class... _UTypes>
inline constexpr bool __tuple_types_same_size<__tuple_types<_Types...>, __tuple_types<_UTypes...>> =
  sizeof...(_Types) == sizeof...(_UTypes);

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

template <class _Types, class _UTypes>
struct __tuple_constructible_struct
{
  static constexpr bool value = __tuple_constructible<_Types, _UTypes>;
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

template <bool _IsImplicitlyConstructible, bool _IsExplicitlyConstructible, bool _IsNothrowConstructible>
struct _TupleConstructorTraits
{
  static constexpr bool __implicit_constructible = _IsImplicitlyConstructible;
  static constexpr bool __explicit_constructible = _IsExplicitlyConstructible;
  static constexpr bool __nothrow_constructible  = _IsNothrowConstructible;
};

using _InvalidTupleConstructor = _TupleConstructorTraits<false, false, false>;

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

template <class... _Types>
struct __tuple_constraints
{
  template <int = 0>
  [[nodiscard]] static _CCCL_API _CCCL_CONSTEVAL auto __check_default_constructible() noexcept
  {
    if constexpr ((is_default_constructible_v<_Types> && ...))
    { // [tuple.cnstr]-6: is_default_constructible_v<Types> is true for all i.
      constexpr bool __is_implicit = (__is_implicitly_default_constructible<_Types>::value && ...);
      constexpr bool __is_nothrow  = (is_nothrow_default_constructible_v<_Types> && ...);
      return _TupleConstructorTraits<__is_implicit, !__is_implicit, __is_nothrow>{};
    }
    else
    {
      return _InvalidTupleConstructor{};
    }
  }

  template <int = 0>
  [[nodiscard]] static _CCCL_API _CCCL_CONSTEVAL auto __check_variadic_copy_constructible() noexcept
  {
    if constexpr (sizeof...(_Types) >= 1 && (is_copy_constructible_v<_Types> && ...))
    { // [tuple.cnstr]-9: sizeof...(Types)  ≥ 1 and is_copy_constructible_v<Types> is true for all i.
      constexpr bool __is_implicit = (is_convertible_v<const _Types&, _Types> && ...);
      constexpr bool __is_nothrow  = (is_nothrow_copy_constructible_v<_Types> && ...);
      return _TupleConstructorTraits<__is_implicit, !__is_implicit, __is_nothrow>{};
    }
    else
    {
      return _InvalidTupleConstructor{};
    }
  }

  template <class... _UTypes>
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL bool __disambiguating_constraints() noexcept
  {
    if constexpr (sizeof...(_Types) == 1)
    { // [tuple.cnstr]-12.1: negation<is_same<remove_cvref_t<U0>, tuple>> if sizeof...(Types) is 1
      using _U0 = __type_index_c<0, _UTypes...>;
      return !is_same_v<remove_cvref_t<_U0>, tuple<_Types...>>;
    }
    else if constexpr (sizeof...(_Types) == 2 || sizeof...(_Types) == 3)
    { // [tuple.cnstr]-12.2: otherwise, if sizeof...(Types) is 2 or 3
      //    !is_same_v<remove_cvref_t<U0>, allocator_arg_t> || is_same_v<remove_cvref_t<T0>, allocator_arg_t>>
      using _U0 = __type_index_c<0, _UTypes...>;
      using _T0 = __type_index_c<0, _Types...>;
      return !is_same_v<remove_cvref_t<_U0>, allocator_arg_t> || is_same_v<remove_cvref_t<_T0>, allocator_arg_t>;
    }
    else
    { // 12.3: otherwise, true_type
      return true;
    }
  }

  template <class... _UTypes>
  [[nodiscard]] static _CCCL_API _CCCL_CONSTEVAL auto __check_variadic_constructible() noexcept
  {
    if constexpr (sizeof...(_Types) != sizeof...(_UTypes))
    { // [tuple.cnstr]-13.1: sizeof...(Types) equals sizeof...(UTypes),
      return _InvalidTupleConstructor{};
    }
    else if constexpr (sizeof...(_Types) == 0)
    { // [tuple.cnstr]-13.2: sizeof...(Types) >= 1,
      return _InvalidTupleConstructor{};
    }
    else if constexpr (!__disambiguating_constraints<_UTypes...>())
    { // [tuple.cnstr]-13.3: disambiguating-constraint is true
      return _InvalidTupleConstructor{};
    }
    else if constexpr ((is_constructible_v<_Types, _UTypes> && ...))
    { // [tuple.cnstr]-13.3: is_constructible<Types, UTypes>... is true
      constexpr bool __is_implicit = (is_convertible_v<_UTypes, _Types> && ...);
      constexpr bool __is_nothrow  = (is_nothrow_constructible_v<_Types, _UTypes> && ...);
      return _TupleConstructorTraits<__is_implicit, !__is_implicit, __is_nothrow>{};
    }
    else
    {
      return _InvalidTupleConstructor{};
    }
  }

  // Get the constraints for the constructor with less rank
  template <class... _UTypes>
  [[nodiscard]] static _CCCL_API _CCCL_CONSTEVAL auto __get_sub_constraints(__tuple_types<_UTypes...>) noexcept
    -> __tuple_constraints<_UTypes...>;

  template <class... _UTypes>
  [[nodiscard]] static _CCCL_API _CCCL_CONSTEVAL auto __check_variadic_constructible_less_rank() noexcept
  {
    if constexpr (sizeof...(_UTypes) == 0)
    {
      return _InvalidTupleConstructor{};
    }
    else if constexpr (sizeof...(_UTypes) < sizeof...(_Types))
    {
      using __constraints_with_arg =
        decltype(__get_sub_constraints(__make_tuple_types_t<__tuple_types<_Types...>, sizeof...(_UTypes)>{}));
      using __constraints_defaulted = decltype(__get_sub_constraints(
        __make_tuple_types_t<__tuple_types<_Types...>, sizeof...(_Types), sizeof...(_UTypes)>{}));

      // The constructor is always explicit.
      constexpr bool __is_arg_constructible =
        decltype(__constraints_with_arg::template __check_variadic_constructible<_UTypes...>())::__implicit_constructible
        || decltype(__constraints_with_arg::template __check_variadic_constructible<
                    _UTypes...>())::__explicit_constructible;
      constexpr bool __rest_is_default_constructible =
        decltype(__constraints_defaulted::__check_default_constructible())::__implicit_constructible
        || decltype(__constraints_defaulted::__check_default_constructible())::__explicit_constructible;
      constexpr bool __is_nothrow =
        decltype(__constraints_with_arg::template __check_variadic_constructible<_UTypes...>())::__nothrow_constructible
        && decltype(__constraints_defaulted::__check_default_constructible())::__nothrow_constructible;
      return _TupleConstructorTraits<false, __is_arg_constructible && __rest_is_default_constructible, __is_nothrow>{};
    }
    else
    {
      return _InvalidTupleConstructor{};
    }
  }

  template <class _Tuple>
  struct __valid_tuple_like_constraints
  {
    static constexpr bool __implicit_constructible =
      __tuple_constructible<_Tuple, __tuple_types<_Types...>> && __tuple_convertible<_Tuple, __tuple_types<_Types...>>;

    static constexpr bool __explicit_constructible =
      __tuple_constructible<_Tuple, __tuple_types<_Types...>> && !__tuple_convertible<_Tuple, __tuple_types<_Types...>>;
  };

  template <class _Tuple>
  struct __valid_tuple_like_constraints_rank_one
  {
    template <class _Tuple2>
    struct _PreferTupleLikeConstructorImpl
        : _Or<
            // Don't attempt the two checks below if the tuple we are given
            // has the same type as this tuple.
            _IsSame<remove_cvref_t<_Tuple2>, tuple<_Types...>>,
            _Lazy<_And, _Not<is_constructible<_Types..., _Tuple2>>, _Not<is_convertible<_Tuple2, _Types...>>>>
    {};

    // This trait is used to disable the tuple-like constructor when
    // the UTypes... constructor should be selected instead.
    // See LWG issue #2549.
    template <class _Tuple2>
    using _PreferTupleLikeConstructor = _PreferTupleLikeConstructorImpl<_Tuple2>;

    static constexpr bool __implicit_constructible =
      __tuple_constructible<_Tuple, __tuple_types<_Types...>> && __tuple_convertible<_Tuple, __tuple_types<_Types...>>
      && _PreferTupleLikeConstructor<_Tuple>::value;

    static constexpr bool __explicit_constructible =
      __tuple_constructible<_Tuple, __tuple_types<_Types...>> && !__tuple_convertible<_Tuple, __tuple_types<_Types...>>
      && _PreferTupleLikeConstructor<_Tuple>::value;
  };

  template <class _Tuple>
  using __tuple_like_constraints =
    conditional_t<sizeof...(_Types) == 1,
                  __valid_tuple_like_constraints_rank_one<_Tuple>,
                  __valid_tuple_like_constraints<_Tuple>>;

  template <class... _UTypes>
  struct __comparison
  {
    static constexpr bool __equality_comparable = (__is_cpp17_equality_comparable_v<_Types, _UTypes> && ...);
    static constexpr bool __nothrow_equality_comparable =
      (__is_cpp17_nothrow_equality_comparable_v<_Types, _UTypes> && ...);

    static constexpr bool __less_than_comparable = (__is_cpp17_less_than_comparable_v<_Types, _UTypes> && ...);
    static constexpr bool __nothrow_less_than_comparable =
      (__is_cpp17_nothrow_less_than_comparable_v<_Types, _UTypes> && ...);
  };
};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TUPLE_TUPLE_CONSTRAINTS_H
