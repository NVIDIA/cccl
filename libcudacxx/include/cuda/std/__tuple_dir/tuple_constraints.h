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

#include <cuda/std/__concepts/different_from.h>
#include <cuda/std/__fwd/get.h>
#include <cuda/std/__fwd/pair.h>
#include <cuda/std/__fwd/subrange.h>
#include <cuda/std/__fwd/tuple.h>
#include <cuda/std/__memory/allocator_arg_t.h>
#include <cuda/std/__tuple_dir/make_tuple_types.h>
#include <cuda/std/__tuple_dir/tuple_element.h>
#include <cuda/std/__tuple_dir/tuple_indices.h>
#include <cuda/std/__tuple_dir/tuple_like.h>
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
#include <cuda/std/__type_traits/is_move_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_assignable.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_convertible.h>
#include <cuda/std/__type_traits/is_nothrow_copy_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_default_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/lazy.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__utility/declval.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class... _Types>
inline constexpr bool __tuple_all_copy_constructible_v = (is_copy_constructible_v<_Types> && ...);

template <class... _Types>
inline constexpr bool __tuple_all_nothrow_copy_constructible_v = (is_nothrow_copy_constructible_v<_Types> && ...);

template <class... _Types>
inline constexpr bool __tuple_all_move_constructible_v = (is_move_constructible_v<_Types> && ...);

template <class... _Types>
inline constexpr bool __tuple_all_nothrow_move_constructible_v = (is_nothrow_move_constructible_v<_Types> && ...);

template <class... _Types>
inline constexpr bool __tuple_all_copy_assignable_v = (is_copy_assignable_v<_Types> && ...);

template <class... _Types>
inline constexpr bool __tuple_all_move_assignable_v = (is_move_assignable_v<_Types> && ...);

// Traits forwarding to `__tuple_types`
template <class, class>
inline constexpr bool __tuple_types_same_size = false;

template <class... _Types, class... _UTypes>
inline constexpr bool __tuple_types_same_size<__tuple_types<_Types...>, __tuple_types<_UTypes...>> =
  sizeof...(_Types) == sizeof...(_UTypes);

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
template <class _Tuple, size_t _ExpectedSize, bool = __tuple_like<_Tuple>>
inline constexpr bool __tuple_like_with_size = false;

template <class _Tuple, size_t _ExpectedSize>
inline constexpr bool __tuple_like_with_size<_Tuple, _ExpectedSize, true> =
  _ExpectedSize == tuple_size<remove_cvref_t<_Tuple>>::value;

struct _InvalidTupleConstructor
{
  static constexpr bool __implicit_construction = false;
  static constexpr bool __explicit_construction = false;
  static constexpr bool __nothrow_construction  = false;
};

enum class __select_constructible
{
  __not_constructible,
  __implicit_constructible,
  __explicit_constructible,
};

template <__select_constructible _Constraint>
inline constexpr bool __select_implicit = _Constraint == __select_constructible::__implicit_constructible;
template <__select_constructible _Constraint>
inline constexpr bool __select_explicit = _Constraint == __select_constructible::__explicit_constructible;

template <class... _Types>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL __select_constructible
__tuple_select_default_constructible(__tuple_types<_Types...>) noexcept
{
  if constexpr (!(is_default_constructible_v<_Types> && ...))
  {
    return __select_constructible::__not_constructible;
  }
  else if constexpr ((__is_implicitly_default_constructible<_Types>::value && ...))
  {
    return __select_constructible::__implicit_constructible;
  }
  else
  {
    return __select_constructible::__explicit_constructible;
  }
}

template <class... _Types>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL __select_constructible
__tuple_select_variadic_copy_constructible(__tuple_types<_Types...>) noexcept
{
  if constexpr (!(is_copy_constructible_v<_Types> && ...))
  {
    return __select_constructible::__not_constructible;
  }
  else if constexpr ((is_convertible_v<const _Types&, _Types> && ...))
  {
    return __select_constructible::__implicit_constructible;
  }
  else
  {
    return __select_constructible::__explicit_constructible;
  }
}

template <class... _Types>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL __select_constructible
__tuple_select_variadic_move_constructible(__tuple_types<_Types...>) noexcept
{
  if constexpr (!(is_move_constructible_v<_Types> && ...))
  {
    return __select_constructible::__not_constructible;
  }
  else if constexpr ((is_convertible_v<_Types&&, _Types> && ...))
  {
    return __select_constructible::__implicit_constructible;
  }
  else
  {
    return __select_constructible::__explicit_constructible;
  }
}

template <class... _Types, class... _UTypes>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL __select_constructible
__tuple_select_variadic_constructible(__tuple_types<_Types...>, __tuple_types<_UTypes...>) noexcept
{
  if constexpr (sizeof...(_Types) != sizeof...(_UTypes))
  { // [tuple.cnstr]-13.1: sizeof...(Types) equals sizeof...(UTypes),
    return __select_constructible::__not_constructible;
  }
  else if constexpr (sizeof...(_Types) == 0)
  { // [tuple.cnstr]-13.2: sizeof...(Types) >= 1,
    return __select_constructible::__not_constructible;
  }
  else if constexpr (sizeof...(_Types) == 2 || sizeof...(_Types) == 3)
  { // [tuple.cnstr]-12.2: otherwise, if sizeof...(Types) is 2 or 3
    //    !is_same_v<remove_cvref_t<U0>, allocator_arg_t> || is_same_v<remove_cvref_t<T0>, allocator_arg_t>>
    using _U0 = __type_index_c<0, _UTypes...>;
    using _T0 = __type_index_c<0, _Types...>;
    if constexpr (!is_same_v<remove_cvref_t<_U0>, allocator_arg_t> || is_same_v<remove_cvref_t<_T0>, allocator_arg_t>)
    { // [tuple.cnstr]-13.3: is_constructible<Types, UTypes>... is true
      if constexpr ((is_constructible_v<_Types, _UTypes> && ...))
      {
        constexpr bool __select_implicit = (is_convertible_v<_UTypes, _Types> && ...);
        return __select_implicit ? __select_constructible::__implicit_constructible
                                 : __select_constructible::__explicit_constructible;
      }
      else
      {
        return __select_constructible::__not_constructible;
      }
    }
    else
    {
      return __select_constructible::__not_constructible;
    }
  }
  else if constexpr ((is_constructible_v<_Types, _UTypes> && ...))
  { // [tuple.cnstr]-13.3: is_constructible<Types, UTypes>... is true
    constexpr bool __select_implicit = (is_convertible_v<_UTypes, _Types> && ...);
    return __select_implicit ? __select_constructible::__implicit_constructible
                             : __select_constructible::__explicit_constructible;
  }
  else
  {
    return __select_constructible::__not_constructible;
  }
}

template <class _Type, class _UType>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL __select_constructible
__tuple_select_variadic_constructible(__tuple_types<_Type>, __tuple_types<_UType>) noexcept
{
  if constexpr (is_same_v<remove_cvref_t<_UType>, tuple<_Type>>)
  { // [tuple.cnstr]-12.1: negation<is_same<remove_cvref_t<U0>, tuple>> if sizeof...(Types) is 1
    return __select_constructible::__not_constructible;
  }
  else if constexpr (!is_constructible_v<_Type, _UType>)
  { // [tuple.cnstr]-13.3: is_constructible<Types, UTypes>... is true
    return __select_constructible::__not_constructible;
  }
  else
  { // [tuple.cnstr]-15: !conjunction_v<is_convertible<UTypes, Types>...>
    return is_convertible_v<_UType, _Type>
           ? __select_constructible::__implicit_constructible
           : __select_constructible::__explicit_constructible;
  }
}

template <class... _Types, class... _UTypes>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL __select_constructible
__tuple_select_variadic_constructible_less_rank(__tuple_types<_Types...>, __tuple_types<_UTypes...>) noexcept
{
  if constexpr (!(sizeof...(_UTypes) < sizeof...(_Types)))
  {
    return __select_constructible::__not_constructible;
  }
  else
  {
    using __arg_list       = __make_tuple_types_t<__tuple_types<_Types...>, sizeof...(_UTypes)>;
    using __defaulted_list = __make_tuple_types_t<__tuple_types<_Types...>, sizeof...(_Types), sizeof...(_UTypes)>;
    if constexpr (::cuda::std::__tuple_select_variadic_constructible(__arg_list{}, __tuple_types<_UTypes...>{})
                  == __select_constructible::__not_constructible)
    {
      return __select_constructible::__not_constructible;
    }
    else if constexpr (::cuda::std::__tuple_select_default_constructible(__defaulted_list{})
                       == __select_constructible::__not_constructible)
    {
      return __select_constructible::__not_constructible;
    }
    else
    {
      return __select_constructible::__explicit_constructible;
    }
  }
}

_CCCL_EXEC_CHECK_DISABLE
template <class _UTuple, class... _Types, size_t... _Indices>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL __select_constructible
__tuple_select_tuple_like_constructible(__tuple_types<_Types...>, __tuple_indices<_Indices...>) noexcept
{
  using ::cuda::std::get;
  if constexpr (__is_cuda_std_ranges_subrange_v<remove_cvref_t<_UTuple>>)
  { // [tuple#cnstr]-29.2: remove_cvref_t<UTuple> is not a specialization of ranges​::​subrange,
    return __select_constructible::__not_constructible;
  }
  else if constexpr (sizeof...(_Types) == 0)
  { // Avoids issues with the size 1 constructor below
    return __select_constructible::__not_constructible;
  }
  else if constexpr (!__tuple_like_with_size<_UTuple, sizeof...(_Types)>)
  { // [tuple#cnstr]-21.1: sizeof...(Types) equals sizeof...(UTypes), and
    // [tuple#cnstr]-25.1: sizeof...(Types) is 2,
    // [tuple#cnstr]-29.3: sizeof...(Types) equals sizeof...(UTypes), and
    return __select_constructible::__not_constructible;
  }
  else if constexpr ((sizeof...(_Types) == 1) && __is_cuda_std_tuple<remove_cvref_t<_UTuple>>
                     && (is_same_v<_Types, tuple_element_t<_Indices, remove_cvref_t<_UTuple>>> && ...))
  { // [tuple#cnstr]-21.3: either sizeof...(Types) is not 1
    // [tuple#cnstr]-21.3: is_same_v<T, U> is false
    return __select_constructible::__not_constructible;
  }
  else if constexpr ((sizeof...(_Types) == 1) && (is_constructible_v<_Types, _UTuple> && ...))
  { // [tuple#cnstr]-21.3: either sizeof...(Types) is not 1, or is_constructible_v<T, _UTuple> are false
    // [tuple#cnstr]-29.5: either sizeof...(Types) is not 1, or is_constructible_v<T, _UTuple> are false
    return __select_constructible::__not_constructible;
  }
  else if constexpr ((sizeof...(_Types) == 1) && (is_convertible_v<_UTuple, _Types> && ...))
  { // [tuple#cnstr]-21.3: either sizeof...(Types) is not 1, or is_convertible_v<_UTuple, T> are false
    // [tuple#cnstr]-29.5: either sizeof...(Types) is not 1, or is_convertible_v<_UTuple, T> are false
    return __select_constructible::__not_constructible;
  }
  else if constexpr (!(is_constructible_v<_Types, decltype(get<_Indices>(::cuda::std::declval<_UTuple>()))> && ...))
  { // [tuple.cnstr]-21.2: is_constructible<Types, decltype(get<I>(std​::​forward<UTuple>(u)))>... is true
    // [tuple.cnstr]-25.2: is_constructible<Types, decltype(get<I>(std​::​forward<UTuple>(u)))>... is true
    // [tuple.cnstr]-29.4: is_constructible<Types, decltype(get<I>(std​::​forward<UTuple>(u)))>... is true
    return __select_constructible::__not_constructible;
  }
  else
  { // [tuple.cnstr]-15: The expression inside explicit is equivalent to:
    // [tuple.cnstr]-23: The expression inside explicit is equivalent to:
    // !(is_convertible_v<decltype(get<I>(FWD(u))), Types> && ...)
    return (is_convertible_v<decltype(get<_Indices>(::cuda::std::declval<_UTuple>())), _Types> && ...)
           ? __select_constructible::__implicit_constructible
           : __select_constructible::__explicit_constructible;
  }
}

struct _InvalidTupleComparison
{
  static constexpr bool __equality_comparable         = false;
  static constexpr bool __nothrow_equality_comparable = false;

  static constexpr bool __less_than_comparable         = false;
  static constexpr bool __nothrow_less_than_comparable = false;
};

template <class, class>
struct _TupleComparableTraits;

template <class... _Types, class... _UTypes>
struct _TupleComparableTraits<__tuple_types<_Types...>, __tuple_types<_UTypes...>>
{
  static constexpr bool __equality_comparable = (__is_cpp17_equality_comparable_v<_Types, _UTypes> && ...);
  static constexpr bool __nothrow_equality_comparable =
    (__is_cpp17_nothrow_equality_comparable_v<_Types, _UTypes> && ...);

  static constexpr bool __less_than_comparable = (__is_cpp17_less_than_comparable_v<_Types, _UTypes> && ...);
  static constexpr bool __nothrow_less_than_comparable =
    (__is_cpp17_nothrow_less_than_comparable_v<_Types, _UTypes> && ...);
};

template <class... _Types, class... _UTypes, enable_if_t<(sizeof...(_Types) == sizeof...(_UTypes)), int> = 0>
[[nodiscard]]
_CCCL_API _CCCL_CONSTEVAL auto __tuple_is_comparable(__tuple_types<_Types...>, __tuple_types<_UTypes...>) noexcept
  -> _TupleComparableTraits<__tuple_types<_Types...>, __tuple_types<_UTypes...>>;
template <class>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL auto __tuple_is_comparable(...) noexcept -> _InvalidTupleComparison;

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TUPLE_TUPLE_CONSTRAINTS_H
