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
#include <cuda/std/__type_traits/is_nothrow_copy_assignable.h>
#include <cuda/std/__type_traits/is_nothrow_copy_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_default_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_move_assignable.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/lazy.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__utility/declval.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class... _Types>
inline constexpr bool __tuple_all_nothrow_copy_constructible_v = (is_nothrow_copy_constructible_v<_Types> && ...);

template <class... _Types>
inline constexpr bool __tuple_all_nothrow_move_constructible_v = (is_nothrow_move_constructible_v<_Types> && ...);

template <class, class>
inline constexpr bool __tuple_all_nothrow_constructible_v = false;

template <class... _Types, class... _UTypes>
inline constexpr bool __tuple_all_nothrow_constructible_v<__tuple_types<_Types...>, __tuple_types<_UTypes...>> =
  (is_nothrow_constructible_v<_Types, _UTypes> && ...);

template <class... _Types>
inline constexpr bool __tuple_all_copy_assignable_v = (is_copy_assignable_v<_Types> && ...);

template <class... _Types>
inline constexpr bool __tuple_all_move_assignable_v = (is_move_assignable_v<_Types> && ...);

// __tuple_like_with_size
template <class _Tuple, size_t _ExpectedSize, bool = __tuple_like<_Tuple>>
inline constexpr bool __tuple_like_with_size = false;

template <class _Tuple, size_t _ExpectedSize>
inline constexpr bool __tuple_like_with_size<_Tuple, _ExpectedSize, true> =
  _ExpectedSize == tuple_size<remove_cvref_t<_Tuple>>::value;

//! @brief Determines whether a constructor is valid and whether it is implicit or explicit
enum class __select_constructor
{
  __none, //!< The constructor is not valid
  __implicit, //!< The constructor is valid and implicit
  __explicit, //!< The constructor is valid and explicit
};

template <__select_constructor _Trait>
inline constexpr bool __can_construct_implicitly = _Trait == __select_constructor::__implicit;
template <__select_constructor _Trait>
inline constexpr bool __can_construct_explicitly = _Trait == __select_constructor::__explicit;
template <__select_constructor _Trait>
inline constexpr bool __can_construct = _Trait != __select_constructor::__none;

template <class... _Types>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL __select_constructor
__tuple_select_default_constructible(__tuple_types<_Types...>) noexcept
{
  if constexpr (!(is_default_constructible_v<_Types> && ...))
  {
    return __select_constructor::__none;
  }
  else if constexpr ((__is_implicitly_default_constructible<_Types>::value && ...))
  {
    return __select_constructor::__implicit;
  }
  else
  {
    return __select_constructor::__explicit;
  }
}

template <class... _Types>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL __select_constructor
__tuple_select_variadic_copy_constructible(__tuple_types<_Types...>) noexcept
{
  if constexpr (!(is_copy_constructible_v<_Types> && ...))
  {
    return __select_constructor::__none;
  }
  else if constexpr ((is_convertible_v<const _Types&, _Types> && ...))
  {
    return __select_constructor::__implicit;
  }
  else
  {
    return __select_constructor::__explicit;
  }
}

template <class _TupleTypes>
inline constexpr __select_constructor __tuple_select_variadic_copy_constructible_v =
  ::cuda::std::__tuple_select_variadic_copy_constructible(_TupleTypes{});

template <class... _Types>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL __select_constructor
__tuple_select_variadic_move_constructible(__tuple_types<_Types...>) noexcept
{
  if constexpr (!(is_move_constructible_v<_Types> && ...))
  {
    return __select_constructor::__none;
  }
  else if constexpr ((is_convertible_v<_Types&&, _Types> && ...))
  {
    return __select_constructor::__implicit;
  }
  else
  {
    return __select_constructor::__explicit;
  }
}

template <class _TupleTypes>
inline constexpr __select_constructor __tuple_select_variadic_move_constructible_v =
  ::cuda::std::__tuple_select_variadic_move_constructible(_TupleTypes{});

template <class... _Types, class... _UTypes>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL __select_constructor
__tuple_select_variadic_constructible(__tuple_types<_Types...>, __tuple_types<_UTypes...>) noexcept
{
  if constexpr (sizeof...(_Types) != sizeof...(_UTypes))
  { // [tuple.cnstr]-13.1: sizeof...(Types) equals sizeof...(UTypes),
    return __select_constructor::__none;
  }
  else if constexpr (sizeof...(_Types) == 0)
  { // [tuple.cnstr]-13.2: sizeof...(Types) >= 1,
    return __select_constructor::__none;
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
        constexpr bool __can_construct_implicitly = (is_convertible_v<_UTypes, _Types> && ...);
        return __can_construct_implicitly ? __select_constructor::__implicit : __select_constructor::__explicit;
      }
      else
      {
        return __select_constructor::__none;
      }
    }
    else
    {
      return __select_constructor::__none;
    }
  }
  else if constexpr ((is_constructible_v<_Types, _UTypes> && ...))
  { // [tuple.cnstr]-13.3: is_constructible<Types, UTypes>... is true
    constexpr bool __can_construct_implicitly = (is_convertible_v<_UTypes, _Types> && ...);
    return __can_construct_implicitly ? __select_constructor::__implicit : __select_constructor::__explicit;
  }
  else
  {
    return __select_constructor::__none;
  }
}

template <class _Type, class _UType>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL __select_constructor
__tuple_select_variadic_constructible(__tuple_types<_Type>, __tuple_types<_UType>) noexcept
{
  if constexpr (is_same_v<remove_cvref_t<_UType>, tuple<_Type>>)
  { // [tuple.cnstr]-12.1: negation<is_same<remove_cvref_t<U0>, tuple>> if sizeof...(Types) is 1
    return __select_constructor::__none;
  }
  else if constexpr (!is_constructible_v<_Type, _UType>)
  { // [tuple.cnstr]-13.3: is_constructible<Types, UTypes>... is true
    return __select_constructor::__none;
  }
  else
  { // [tuple.cnstr]-15: !conjunction_v<is_convertible<UTypes, Types>...>
    return is_convertible_v<_UType, _Type> ? __select_constructor::__implicit : __select_constructor::__explicit;
  }
}

template <class _TupleTypes, class _TupleUTypes>
inline constexpr __select_constructor __tuple_select_variadic_constructible_v =
  ::cuda::std::__tuple_select_variadic_constructible(_TupleTypes{}, _TupleUTypes{});

template <class... _Types, class... _UTypes>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL __select_constructor
__tuple_select_variadic_constructible_less_rank(__tuple_types<_Types...>, __tuple_types<_UTypes...>) noexcept
{
  if constexpr (!(sizeof...(_UTypes) < sizeof...(_Types)))
  {
    return __select_constructor::__none;
  }
  else if constexpr (sizeof...(_UTypes) == 0)
  {
    return __select_constructor::__none;
  }
  else
  {
    using __arg_list       = __make_tuple_types_t<__tuple_types<_Types...>, sizeof...(_UTypes)>;
    using __defaulted_list = __make_tuple_types_t<__tuple_types<_Types...>, sizeof...(_Types), sizeof...(_UTypes)>;
    if constexpr (::cuda::std::__tuple_select_variadic_constructible(__arg_list{}, __tuple_types<_UTypes...>{})
                  == __select_constructor::__none)
    {
      return __select_constructor::__none;
    }
    else if constexpr (::cuda::std::__tuple_select_default_constructible(__defaulted_list{})
                       == __select_constructor::__none)
    {
      return __select_constructor::__none;
    }
    else
    {
      return __select_constructor::__explicit;
    }
  }
}

template <class _TupleTypes, class _TupleUTypes>
inline constexpr __select_constructor __tuple_select_variadic_constructible_less_rank_v =
  ::cuda::std::__tuple_select_variadic_constructible_less_rank(_TupleTypes{}, _TupleUTypes{});

_CCCL_EXEC_CHECK_DISABLE
template <class _UTuple, class... _Types, size_t... _Indices>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL __select_constructor
__tuple_select_tuple_like_constructible(__tuple_types<_Types...>, __tuple_indices<_Indices...>) noexcept
{
  using ::cuda::std::get;
  if constexpr (__is_cuda_std_ranges_subrange_v<remove_cvref_t<_UTuple>>)
  { // [tuple#cnstr]-29.2: remove_cvref_t<UTuple> is not a specialization of ranges​::​subrange,
    return __select_constructor::__none;
  }
  else if constexpr (is_same_v<_UTuple, const tuple<_Types...>&> || is_same_v<_UTuple, tuple<_Types...>&&>)
  { // Prefers the copy/move constructor
    return __select_constructor::__none;
  }
  else if constexpr (sizeof...(_Types) == 0)
  { // Avoids issues with the size 1 constructor below
    return __select_constructor::__none;
  }
  else if constexpr (!__tuple_like_with_size<_UTuple, sizeof...(_Types)>)
  { // [tuple#cnstr]-21.1: sizeof...(Types) equals sizeof...(UTypes), and
    // [tuple#cnstr]-25.1: sizeof...(Types) is 2,
    // [tuple#cnstr]-29.3: sizeof...(Types) equals sizeof...(UTypes), and
    return __select_constructor::__none;
  }
  else if constexpr (!(is_constructible_v<_Types, decltype(get<_Indices>(::cuda::std::declval<_UTuple>()))> && ...))
  { // [tuple.cnstr]-21.2: is_constructible<Types, decltype(get<I>(std​::​forward<UTuple>(u)))>... is true
    // [tuple.cnstr]-25.2: is_constructible<Types, decltype(get<I>(std​::​forward<UTuple>(u)))>... is true
    // [tuple.cnstr]-29.4: is_constructible<Types, decltype(get<I>(std​::​forward<UTuple>(u)))>... is true
    return __select_constructor::__none;
  }
  else
  { // [tuple.cnstr]-15: The expression inside explicit is equivalent to:
    // [tuple.cnstr]-23: The expression inside explicit is equivalent to:
    // !(is_convertible_v<decltype(get<I>(FWD(u))), Types> && ...)
    return (is_convertible_v<decltype(get<_Indices>(::cuda::std::declval<_UTuple>())), _Types> && ...)
           ? __select_constructor::__implicit
           : __select_constructor::__explicit;
  }
}

_CCCL_EXEC_CHECK_DISABLE
template <class _UTuple, class _Type, size_t _Index>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL __select_constructor
__tuple_select_tuple_like_constructible(__tuple_types<_Type>, __tuple_indices<_Index>) noexcept
{
  using ::cuda::std::get;
  if constexpr (__is_cuda_std_ranges_subrange_v<remove_cvref_t<_UTuple>>)
  { // [tuple#cnstr]-29.2: remove_cvref_t<UTuple> is not a specialization of ranges​::​subrange,
    return __select_constructor::__none;
  }
  else if constexpr (is_same_v<_UTuple, const tuple<_Type>&> || is_same_v<_UTuple, tuple<_Type>&&>)
  { // Prefers the copy/move constructor
    return __select_constructor::__none;
  }
  else if constexpr (!__tuple_like_with_size<_UTuple, 1>)
  { // [tuple#cnstr]-21.1: sizeof...(Types) equals sizeof...(UTypes), and
    // [tuple#cnstr]-29.3: sizeof...(Types) equals sizeof...(UTypes), and
    return __select_constructor::__none;
  }
  else if constexpr (__is_cuda_std_tuple<remove_cvref_t<_UTuple>>
                     && is_same_v<_Type, tuple_element_t<_Index, remove_cvref_t<_UTuple>>>)
  { // [tuple#cnstr]-21.3: either sizeof...(Types) is not 1
    // [tuple#cnstr]-21.3: is_same_v<T, U> is false
    return __select_constructor::__none;
  }
  else if constexpr (is_constructible_v<_Type, _UTuple>)
  { // [tuple#cnstr]-21.3: either sizeof...(Types) is not 1, or is_constructible_v<T, _UTuple> are false
    // [tuple#cnstr]-29.5: either sizeof...(Types) is not 1, or is_constructible_v<T, _UTuple> are false
    return __select_constructor::__none;
  }
  else if constexpr (is_convertible_v<_UTuple, _Type>)
  { // [tuple#cnstr]-21.3: either sizeof...(Types) is not 1, or is_convertible_v<_UTuple, T> are false
    // [tuple#cnstr]-29.5: either sizeof...(Types) is not 1, or is_convertible_v<_UTuple, T> are false
    return __select_constructor::__none;
  }
  else if constexpr (!is_constructible_v<_Type, decltype(get<_Index>(::cuda::std::declval<_UTuple>()))>)
  { // [tuple.cnstr]-21.2: is_constructible<Types, decltype(get<I>(std​::​forward<UTuple>(u)))>... is true
    // [tuple.cnstr]-25.2: is_constructible<Types, decltype(get<I>(std​::​forward<UTuple>(u)))>... is true
    // [tuple.cnstr]-29.4: is_constructible<Types, decltype(get<I>(std​::​forward<UTuple>(u)))>... is true
    return __select_constructor::__none;
  }
  else
  { // [tuple.cnstr]-15: The expression inside explicit is equivalent to:
    // [tuple.cnstr]-23: The expression inside explicit is equivalent to:
    // !(is_convertible_v<decltype(get<I>(FWD(u))), Types> && ...)
    return is_convertible_v<decltype(get<_Index>(::cuda::std::declval<_UTuple>())), _Type>
           ? __select_constructor::__implicit
           : __select_constructor::__explicit;
  }
}

template <class _UTuple, class _TupleTypes, class _TupleIndices>
inline constexpr __select_constructor __tuple_select_tuple_like_constructible_v =
  ::cuda::std::__tuple_select_tuple_like_constructible<_UTuple>(_TupleTypes{}, _TupleIndices{});

_CCCL_EXEC_CHECK_DISABLE
template <class _UTuple, class _Type, class... _Types, size_t _Index, size_t... _Indices>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL bool
__tuple_nothrow_tuple_like_constructible(__tuple_types<_Type, _Types...>, __tuple_indices<_Index, _Indices...>) noexcept
{
  using ::cuda::std::get;
  if constexpr (!is_nothrow_constructible_v<_Type, decltype(get<_Index>(::cuda::std::declval<_UTuple>()))>)
  {
    return false;
  }
  else if constexpr (sizeof...(_Types) != 0)
  {
    return ::cuda::std::__tuple_nothrow_tuple_like_constructible<_UTuple>(
      __tuple_types<_Types...>{}, __tuple_indices<_Indices...>{});
  }
  else
  {
    return true;
  }
}

template <class _UTuple>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL bool
__tuple_nothrow_tuple_like_constructible(__tuple_types<>, __tuple_indices<>) noexcept
{
  return true;
}

template <class _UTuple, class _TupleTypes, class _TupleIndices>
inline constexpr bool __tuple_nothrow_tuple_like_constructible_v =
  ::cuda::std::__tuple_nothrow_tuple_like_constructible<_UTuple>(_TupleTypes{}, _TupleIndices{});

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

//! @brief Determines whether an assignment is valid and also whether it does not throw
enum class __select_assignment
{
  __none, //!< No assignment possible
  __is_nothrow, //!< Assignment possible, is nothrow
  __may_throw, //!< Assignment possible, may throw
};

template <__select_assignment _Trait>
inline constexpr bool __can_assign = _Trait != __select_assignment::__none;
template <__select_assignment _Trait>
inline constexpr bool __can_nothrow_assign = _Trait == __select_assignment::__is_nothrow;

template <class... _Types>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL __select_assignment
__tuple_select_const_copy_assignable(__tuple_types<_Types...>) noexcept
{
  if constexpr (!(is_copy_assignable_v<const _Types> && ...))
  { // [tuple.assign]-5: is_copy_assignable_v<const Types> is true for all i.
    return __select_assignment::__none;
  }
  else if constexpr ((is_nothrow_copy_assignable_v<const _Types> && ...))
  {
    return __select_assignment::__is_nothrow;
  }
  else
  {
    return __select_assignment::__may_throw;
  }
}

template <class _TupleTypes>
inline constexpr __select_assignment __tuple_select_const_copy_assignable_v =
  ::cuda::std::__tuple_select_const_copy_assignable(_TupleTypes{});

template <class... _Types>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL __select_assignment
__tuple_select_const_move_assignable(__tuple_types<_Types...>) noexcept
{
  if constexpr (!(is_assignable_v<const _Types&, _Types> && ...))
  { // [tuple.assign]-12: is_assignable_v<const Types&, Types> is true for all i.
    return __select_assignment::__none;
  }
  else if constexpr ((is_nothrow_assignable_v<const _Types&, _Types> && ...))
  {
    return __select_assignment::__is_nothrow;
  }
  else
  {
    return __select_assignment::__may_throw;
  }
}

template <class _TupleTypes>
inline constexpr __select_assignment __tuple_select_const_move_assignable_v =
  ::cuda::std::__tuple_select_const_move_assignable(_TupleTypes{});

template <bool _IsConst, class... _Types, class... _UTypes>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL __select_assignment
__tuple_select_converting_assignable(__tuple_types<_Types...>, __tuple_types<_UTypes...>) noexcept
{
  if constexpr (sizeof...(_Types) != sizeof...(_UTypes))
  { // [tuple.assign]-15.1: sizeof...(Types) equals sizeof...(UTypes) and
    return __select_assignment::__none;
  }
  else if constexpr (is_same_v<tuple<_Types...>, tuple<_UTypes...>>)
  { // Disambiguate the non-converting assignments
    return __select_assignment::__none;
  }
  else if constexpr (_IsConst)
  {
    if constexpr ((is_assignable_v<const _Types&, _UTypes> && ...))
    { // [tuple.assign]-18.2: is_assignable_v<const Types&, const UTypes&> is true for all i.
      // [tuple.assign]-24.2: is_assignable_v<const Types&, UTypes> is true for all i.
      return (is_nothrow_assignable_v<const _Types&, _UTypes> && ...)
             ? __select_assignment::__is_nothrow
             : __select_assignment::__may_throw;
    }
    else
    {
      return __select_assignment::__none;
    }
  }
  else if constexpr ((is_assignable_v<_Types&, _UTypes> && ...))
  { // [tuple.assign]-15.2: is_assignable_v<Types&, const UTypes&> is true for all i.
    // [tuple.assign]-21.2: is_assignable_v<Types&, UTypes> is true for all i.
    return (is_nothrow_assignable_v<_Types&, _UTypes> && ...)
           ? __select_assignment::__is_nothrow
           : __select_assignment::__may_throw;
  }
  else
  {
    return __select_assignment::__none;
  }
}

template <bool _IsConst, class _TupleTypes, class _TupleUTypes>
inline constexpr __select_assignment __tuple_select_converting_assignable_v =
  ::cuda::std::__tuple_select_converting_assignable<_IsConst>(_TupleTypes{}, _TupleUTypes{});

_CCCL_EXEC_CHECK_DISABLE
template <bool _IsConst, class _UTuple, class... _Types, size_t... _Indices>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL __select_assignment
__tuple_select_tuple_like_assignable(__tuple_types<_Types...>, __tuple_indices<_Indices...>) noexcept
{
  using ::cuda::std::get;
  if constexpr (is_same_v<remove_cvref_t<_UTuple>, tuple<_Types...>>)
  { // [tuple.assign]-39.1: different-from<UTuple, tuple>
    return __select_assignment::__none;
  }
  else if constexpr (__is_cuda_std_ranges_subrange_v<remove_cvref_t<_UTuple>>)
  { // [tuple.assign]-39.2: remove_cvref_t<UTuple> is not a specialization of ranges​::​subrange,
    return __select_assignment::__none;
  }
  else if constexpr (!__tuple_like_with_size<_UTuple, sizeof...(_Types)>)
  { // [tuple.assign]-39.3: sizeof...(Types) equals tuple_size_v<remove_cvref_t<UTuple>>, and
    return __select_assignment::__none;
  }
  else if constexpr (_IsConst)
  {
    if constexpr ((is_assignable_v<const _Types&, decltype(get<_Indices>(::cuda::std::declval<_UTuple>()))> && ...))
    { // [tuple.assign]-42.4: is_assignable_v<const T_i&, decltype(get<i>(std​::​forward<UTuple>(u)))> is true for
      // all i
      return (is_nothrow_assignable_v<const _Types&, decltype(get<_Indices>(::cuda::std::declval<_UTuple>()))> && ...)
             ? __select_assignment::__is_nothrow
             : __select_assignment::__may_throw;
    }
    else
    {
      return __select_assignment::__none;
    }
  }
  else if constexpr ((is_assignable_v<_Types&, decltype(get<_Indices>(::cuda::std::declval<_UTuple>()))> && ...))
  { // [tuple.assign]-39.4: is_assignable_v<T_i&, decltype(get<i>(std​::​forward<UTuple>(u)))> is true for all i
    return (is_nothrow_assignable_v<_Types&, decltype(get<_Indices>(::cuda::std::declval<_UTuple>()))> && ...)
           ? __select_assignment::__is_nothrow
           : __select_assignment::__may_throw;
  }
  else
  {
    return __select_assignment::__none;
  }
}

template <bool _IsConst, class _UTuple, class _TupleTypes, class _TupleIndices>
inline constexpr __select_assignment __tuple_select_tuple_like_assignable_v =
  ::cuda::std::__tuple_select_tuple_like_assignable<_IsConst, _UTuple>(_TupleTypes{}, _TupleIndices{});

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TUPLE_TUPLE_CONSTRAINTS_H
