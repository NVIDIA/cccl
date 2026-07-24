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
#include <cuda/std/__type_traits/reference_constructs_from_temporary.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__type_traits/sfinae_traits.h>
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

template <class... _Types>
struct __tuple_constraints
{
  template <int = 0>
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL __select_constructor __select_default_constructible() noexcept
  {
    if constexpr (!(is_default_constructible_v<_Types> && ...))
    {
      return __select_constructor::__invalid;
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

  template <int = 0>
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL __select_constructor __select_variadic_copy_constructible() noexcept
  {
    if constexpr (!(is_copy_constructible_v<_Types> && ...))
    {
      return __select_constructor::__invalid;
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

  template <int = 0>
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL __select_constructor __select_variadic_move_constructible() noexcept
  {
    if constexpr (!(is_move_constructible_v<_Types> && ...))
    {
      return __select_constructor::__invalid;
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

  template <class... _UTypes>
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL bool __disambiguate_variadic(__tuple_types<_UTypes...>) noexcept
  {
    if constexpr (sizeof...(_Types) == 0 || sizeof...(_UTypes) == 0)
    {
      return false;
    }
    else if constexpr (sizeof...(_UTypes) == 1)
    { // [tuple.cnstr]-12.1: negation<is_same<remove_cvref_t<U0>, tuple>> if sizeof...(Types) is 1
      return !(is_same_v<remove_cvref_t<_UTypes>, tuple<_Types...>> || ...);
    }
    else if constexpr (sizeof...(_UTypes) == 2 && sizeof...(_Types) != 0)
    { // [tuple.cnstr]-12.2: !is_same_v<remove_cvref_t<U0>, allocator_arg_t>
      //                   || is_same_v<remove_cvref_t<T0>, allocator_arg_t>>
      using _U0 = __type_index_c<0, _UTypes...>;
      using _T0 = __type_index_c<0, _Types...>;
      return !is_same_v<remove_cvref_t<_U0>, allocator_arg_t> || is_same_v<remove_cvref_t<_T0>, allocator_arg_t>;
    }
    else
    {
      return true;
    }
  }

  template <class... _UTypes>
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL __select_constructor
  __select_variadic_constructible(__tuple_types<_UTypes...>) noexcept
  {
    // NOLINTBEGIN(bugprone-branch-clone)
    if constexpr (sizeof...(_Types) != sizeof...(_UTypes))
    { // [tuple.cnstr]-13.1: sizeof...(Types) equals sizeof...(UTypes),
      return __select_constructor::__invalid;
    }
    else if constexpr (sizeof...(_Types) == 0)
    { // [tuple.cnstr]-13.2: sizeof...(Types) >= 1,
      return __select_constructor::__invalid;
    }
    else if constexpr (sizeof...(_Types) == 2 || sizeof...(_Types) == 3)
    { // [tuple.cnstr]-12.2: otherwise, if sizeof...(Types) is 2 or 3
      using _U0 = __type_index_c<0, _UTypes...>;
      using _T0 = __type_index_c<0, _Types...>;
      if constexpr (!is_same_v<remove_cvref_t<_U0>, allocator_arg_t> || is_same_v<remove_cvref_t<_T0>, allocator_arg_t>)
      { // [tuple.cnstr]-13.3: !is_same_v<remove_cvref_t<U0>, allocator_arg_t> || is_same_v<remove_cvref_t<T0>,
        // allocator_arg_t>>
        if constexpr (!(is_constructible_v<_Types, _UTypes> && ...))
        { // [tuple.cnstr]-13.3: is_constructible<Types, UTypes>... is true
          return __select_constructor::__invalid;
        }
#if defined(_CCCL_BUILTIN_REFERENCE_CONSTRUCTS_FROM_TEMPORARY)
        else if constexpr ((reference_constructs_from_temporary_v<_Types, _UTypes&&> || ...))
        { // [tuple.cnstr]-15: This constructor is defined as deleted if
          // (reference_constructs_from_temporary_v<Types, UTypes&&> || ...) is true
          return __select_constructor::__deleted;
        }
#endif // _CCCL_BUILTIN_REFERENCE_CONSTRUCTS_FROM_TEMPORARY
        else if constexpr (!(is_convertible_v<_UTypes, _Types> && ...))
        { // [tuple.cnstr]-15: !conjunction_v<is_convertible<UTypes, Types>...>
          return __select_constructor::__explicit;
        }
        else
        {
          return __select_constructor::__implicit;
        }
      }
      else
      {
        return __select_constructor::__invalid;
      }
    }
    else if constexpr (!(is_constructible_v<_Types, _UTypes> && ...))
    { // [tuple.cnstr]-13.3: is_constructible<Types, UTypes>... is true
      return __select_constructor::__invalid;
    }
#if defined(_CCCL_BUILTIN_REFERENCE_CONSTRUCTS_FROM_TEMPORARY)
    else if constexpr ((reference_constructs_from_temporary_v<_Types, _UTypes&&> || ...))
    { // [tuple.cnstr]-15: This constructor is defined as deleted if
      // (reference_constructs_from_temporary_v<Types, UTypes&&> || ...) is true
      return __select_constructor::__deleted;
    }
#endif // _CCCL_BUILTIN_REFERENCE_CONSTRUCTS_FROM_TEMPORARY
    else if constexpr (!(is_convertible_v<_UTypes, _Types> && ...))
    { // [tuple.cnstr]-15: !conjunction_v<is_convertible<UTypes, Types>...>
      return __select_constructor::__explicit;
    }
    else
    {
      return __select_constructor::__implicit;
    }
    // NOLINTEND(bugprone-branch-clone)
  }

  template <class _UType>
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL __select_constructor
  __select_variadic_constructible(__tuple_types<_UType>) noexcept
  {
    // NOLINTBEGIN(bugprone-branch-clone)
    if constexpr (sizeof...(_Types) != 1)
    {
      return __select_constructor::__invalid;
    }
    else if constexpr (__is_tuple_of_iterator_references_v<remove_cvref_t<_UType>>)
    {
      return __select_constructor::__invalid;
    }
    else if constexpr (is_same_v<remove_cvref_t<_UType>, tuple<_Types...>>)
    { // [tuple.cnstr]-12.1: negation<is_same<remove_cvref_t<U0>, tuple>> if sizeof...(Types) is 1
      return __select_constructor::__invalid;
    }
    else if constexpr (!(is_constructible_v<_Types, _UType> && ...))
    { // [tuple.cnstr]-13.3: is_constructible<Types, UTypes>... is true
      return __select_constructor::__invalid;
    }
#if defined(_CCCL_BUILTIN_REFERENCE_CONSTRUCTS_FROM_TEMPORARY)
    else if constexpr ((reference_constructs_from_temporary_v<_Types, _UType&&> || ...))
    { // [tuple.cnstr]-15: This constructor is defined as deleted if
      // (reference_constructs_from_temporary_v<Types, UTypes&&> || ...) is true
      return __select_constructor::__deleted;
    }
#endif // _CCCL_BUILTIN_REFERENCE_CONSTRUCTS_FROM_TEMPORARY
    else if constexpr (!(is_convertible_v<_UType, _Types> && ...))
    { // [tuple.cnstr]-15: !conjunction_v<is_convertible<UTypes, Types>...>
      return __select_constructor::__explicit;
    }
    else
    {
      return __select_constructor::__implicit;
    }
    // NOLINTEND(bugprone-branch-clone)
  }

  template <class... _UTypes>
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL __tuple_constraints<_UTypes...>
  __get_tuple_constraints(__tuple_types<_UTypes...>) noexcept
  {
    return {};
  }

  template <class... _UTypes>
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL __select_constructor
  __select_variadic_constructible_less_rank(__tuple_types<_UTypes...>) noexcept
  {
    // NOLINTBEGIN(bugprone-branch-clone)
    if constexpr (!(sizeof...(_UTypes) < sizeof...(_Types)))
    {
      return __select_constructor::__invalid;
    }
    else if constexpr (sizeof...(_UTypes) == 0)
    {
      return __select_constructor::__invalid;
    }
    else if constexpr (sizeof...(_UTypes) == 1 && (is_same_v<remove_cvref_t<_UTypes>, tuple<_Types...>> && ...))
    { // Avoid this shadowing the copy / move constructors
      return __select_constructor::__invalid;
    }
    else
    { // MSVC has issues with constexpr variables here, so no `__can_construct<_Trait>` or constexpr variable
      using __arg_list        = __make_tuple_types_t<__tuple_types<_Types...>, sizeof...(_UTypes)>;
      using __arg_constraints = decltype(__get_tuple_constraints(__arg_list{}));
      if constexpr (__arg_constraints::__select_variadic_constructible(__tuple_types<_UTypes...>{})
                      == __select_constructor::__invalid
                    || __arg_constraints::__select_variadic_constructible(__tuple_types<_UTypes...>{})
                         == __select_constructor::__deleted)
      {
        return __select_constructor::__invalid;
      }
      else
      {
        using __defaulted_list = __make_tuple_types_t<__tuple_types<_Types...>, sizeof...(_Types), sizeof...(_UTypes)>;
        using __defautled_constraints = decltype(__get_tuple_constraints(__defaulted_list{}));
        if constexpr (__defautled_constraints::__select_default_constructible() == __select_constructor::__invalid
                      || __defautled_constraints::__select_default_constructible() == __select_constructor::__deleted)
        {
          return __select_constructor::__invalid;
        }
        else
        {
          return __select_constructor::__explicit;
        }
      }
    }
    _CCCL_UNREACHABLE();
    // NOLINTEND(bugprone-branch-clone)
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <class _UTuple, size_t... _Indices>
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL __select_constructor
  __select_tuple_like_constructible(__tuple_indices<_Indices...>) noexcept
  {
    // NOLINTBEGIN(bugprone-branch-clone)
    using ::cuda::std::get;
    if constexpr (__is_cuda_std_ranges_subrange_v<remove_cvref_t<_UTuple>>)
    { // [tuple#cnstr]-29.2: remove_cvref_t<UTuple> is not a specialization of ranges::subrange,
      return __select_constructor::__invalid;
    }
    else if constexpr (is_same_v<_UTuple, const tuple<_Types...>&> || is_same_v<_UTuple, tuple<_Types...>&&>)
    { // Prefers the copy/move constructor
      return __select_constructor::__invalid;
    }
    else if constexpr (sizeof...(_Types) == 0)
    { // Avoids issues with the size 1 constructor below
      return __select_constructor::__invalid;
    }
    else if constexpr (!__tuple_like_with_size<_UTuple, sizeof...(_Types)>)
    { // [tuple#cnstr]-21.1: sizeof...(Types) equals sizeof...(UTypes), and
      // [tuple#cnstr]-25.1: sizeof...(Types) is 2,
      // [tuple#cnstr]-29.3: sizeof...(Types) equals sizeof...(UTypes), and
      return __select_constructor::__invalid;
    }
    else if constexpr (!(is_constructible_v<_Types, decltype(get<_Indices>(::cuda::std::declval<_UTuple>()))> && ...))
    { // [tuple.cnstr]-21.2: is_constructible<Types, decltype(get<I>(std::forward<UTuple>(u)))>... is true
      // [tuple.cnstr]-25.2: is_constructible<Types, decltype(get<I>(std::forward<UTuple>(u)))>... is true
      // [tuple.cnstr]-29.4: is_constructible<Types, decltype(get<I>(std::forward<UTuple>(u)))>... is true
      return __select_constructor::__invalid;
    }
#if defined(_CCCL_BUILTIN_REFERENCE_CONSTRUCTS_FROM_TEMPORARY)
    else if constexpr ((reference_constructs_from_temporary_v<_Types,
                                                              decltype(get<_Indices>(::cuda::std::declval<_UTuple>()))>
                        || ...))
    { // [tuple.cnstr]-23: This constructor is defined as deleted if
      // [tuple.cnstr]-27: This constructor is defined as deleted if
      // [tuple.cnstr]-31: This constructor is defined as deleted if
      // (reference_constructs_from_temporary_v<Types, decltype(get<I>(FWD(u)))> || ...) is true
      return __select_constructor::__deleted;
    }
#endif // _CCCL_BUILTIN_REFERENCE_CONSTRUCTS_FROM_TEMPORARY
    else if constexpr (!(is_convertible_v<decltype(get<_Indices>(::cuda::std::declval<_UTuple>())), _Types> && ...))
    { // [tuple.cnstr]-15: The expression inside explicit is equivalent to:
      // [tuple.cnstr]-23: The expression inside explicit is equivalent to:
      // [tuple.cnstr]-31: The expression inside explicit is equivalent to:
      // !(is_convertible_v<decltype(get<I>(FWD(u))), Types> && ...)
      return __select_constructor::__explicit;
    }
    else
    {
      return __select_constructor::__implicit;
    }
    // NOLINTEND(bugprone-branch-clone)
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <class _UTuple, size_t _Index>
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL __select_constructor
  __select_tuple_like_constructible(__tuple_indices<_Index>) noexcept
  {
    // NOLINTBEGIN(bugprone-branch-clone)
    using ::cuda::std::get;
    if constexpr (__is_cuda_std_ranges_subrange_v<remove_cvref_t<_UTuple>>)
    { // [tuple#cnstr]-29.2: remove_cvref_t<UTuple> is not a specialization of ranges::subrange,
      return __select_constructor::__invalid;
    }
    else if constexpr (is_same_v<_UTuple, const tuple<_Types...>&> || is_same_v<_UTuple, tuple<_Types...>&&>)
    { // Prefers the copy/move constructor
      return __select_constructor::__invalid;
    }
    else if constexpr (!__tuple_like_with_size<_UTuple, 1>)
    { // [tuple#cnstr]-21.1: sizeof...(Types) equals sizeof...(UTypes), and
      // [tuple#cnstr]-29.3: sizeof...(Types) equals sizeof...(UTypes), and
      return __select_constructor::__invalid;
    }
    else if constexpr (__is_cuda_std_tuple<remove_cvref_t<_UTuple>>
                       && (is_same_v<_Types, tuple_element_t<_Index, remove_cvref_t<_UTuple>>> && ...))
    { // [tuple#cnstr]-21.3: either sizeof...(Types) is not 1
      // [tuple#cnstr]-21.3: is_same_v<T, U> is false
      return __select_constructor::__invalid;
    }
    else if constexpr ((is_constructible_v<_Types, _UTuple> && ...))
    { // [tuple#cnstr]-21.3: either sizeof...(Types) is not 1, or is_constructible_v<T, _UTuple> are false
      // [tuple#cnstr]-29.5: either sizeof...(Types) is not 1, or is_constructible_v<T, _UTuple> are false
      return __select_constructor::__invalid;
    }
    else if constexpr ((is_convertible_v<_UTuple, _Types> && ...))
    { // [tuple#cnstr]-21.3: either sizeof...(Types) is not 1, or is_convertible_v<_UTuple, T> are false
      // [tuple#cnstr]-29.5: either sizeof...(Types) is not 1, or is_convertible_v<_UTuple, T> are false
      return __select_constructor::__invalid;
    }
    else if constexpr (!(is_constructible_v<_Types, decltype(get<_Index>(::cuda::std::declval<_UTuple>()))> && ...))
    { // [tuple.cnstr]-21.2: is_constructible<Types, decltype(get<I>(std::forward<UTuple>(u)))>... is true
      // [tuple.cnstr]-29.4: is_constructible<Types, decltype(get<I>(std::forward<UTuple>(u)))>... is true
      return __select_constructor::__invalid;
    }
#if defined(_CCCL_BUILTIN_REFERENCE_CONSTRUCTS_FROM_TEMPORARY)
    else if constexpr ((reference_constructs_from_temporary_v<_Types,
                                                              decltype(get<_Index>(::cuda::std::declval<_UTuple>()))>
                        || ...))
    { // [tuple.cnstr]-23: This constructor is defined as deleted if
      // [tuple.cnstr]-31: This constructor is defined as deleted if
      // (reference_constructs_from_temporary_v<Types, decltype(get<I>(FWD(u)))> || ...) is true
      return __select_constructor::__deleted;
    }
#endif // _CCCL_BUILTIN_REFERENCE_CONSTRUCTS_FROM_TEMPORARY
    else if constexpr (!(is_convertible_v<decltype(get<_Index>(::cuda::std::declval<_UTuple>())), _Types> && ...))
    { // [tuple.cnstr]-15: The expression inside explicit is equivalent to:
      // [tuple.cnstr]-23: The expression inside explicit is equivalent to:
      // !(is_convertible_v<decltype(get<I>(FWD(u))), Types> && ...)
      return __select_constructor::__explicit;
    }
    else
    {
      return __select_constructor::__implicit;
    }
    // NOLINTEND(bugprone-branch-clone)
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <class _UTuple, size_t... _Indices>
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL bool
  __tuple_nothrow_tuple_like_constructible(__tuple_indices<_Indices...>) noexcept
  {
    using ::cuda::std::get;
    return (is_nothrow_constructible_v<_Types, decltype(get<_Indices>(::cuda::std::declval<_UTuple>()))> && ...);
  }

  template <class _UTuple>
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL bool
  __tuple_nothrow_tuple_like_constructible(__tuple_indices<>) noexcept
  {
    return true;
  }

  template <int = 0>
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL __select_assignment __select_const_copy_assignable() noexcept
  {
    if constexpr (!(is_copy_assignable_v<const _Types> && ...))
    { // [tuple.assign]-5: is_copy_assignable_v<const Types> is true for all i.
      return __select_assignment::__invalid;
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

  template <int = 0>
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL __select_assignment __select_const_move_assignable() noexcept
  {
    if constexpr (!(is_assignable_v<const _Types&, _Types> && ...))
    { // [tuple.assign]-12: is_assignable_v<const Types&, Types> is true for all i.
      return __select_assignment::__invalid;
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

  template <bool _IsConst, class... _UTypes>
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL __select_assignment
  __select_converting_assignable(__tuple_types<_UTypes...>) noexcept
  {
    // NOLINTBEGIN(bugprone-branch-clone)
    if constexpr (sizeof...(_Types) != sizeof...(_UTypes))
    { // [tuple.assign]-15.1: sizeof...(Types) equals sizeof...(UTypes) and
      return __select_assignment::__invalid;
    }
    else if constexpr (is_same_v<tuple<_Types...>, tuple<_UTypes...>>)
    { // Disambiguate the non-converting assignments
      return __select_assignment::__invalid;
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
        return __select_assignment::__invalid;
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
      return __select_assignment::__invalid;
    }
    // NOLINTEND(bugprone-branch-clone)
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <bool _IsConst, class _UTuple, size_t... _Indices>
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL __select_assignment
  __select_tuple_like_assignable(__tuple_indices<_Indices...>) noexcept
  {
    // NOLINTBEGIN(bugprone-branch-clone)
    using ::cuda::std::get;
    if constexpr (is_same_v<remove_cvref_t<_UTuple>, tuple<_Types...>>)
    { // [tuple.assign]-39.1: different-from<UTuple, tuple>
      return __select_assignment::__invalid;
    }
    else if constexpr (__is_cuda_std_ranges_subrange_v<remove_cvref_t<_UTuple>>)
    { // [tuple.assign]-39.2: remove_cvref_t<UTuple> is not a specialization of ranges::subrange,
      return __select_assignment::__invalid;
    }
    else if constexpr (!__tuple_like_with_size<_UTuple, sizeof...(_Types)>)
    { // [tuple.assign]-39.3: sizeof...(Types) equals tuple_size_v<remove_cvref_t<UTuple>>, and
      return __select_assignment::__invalid;
    }
    else if constexpr (_IsConst)
    {
      if constexpr ((is_assignable_v<const _Types&, decltype(get<_Indices>(::cuda::std::declval<_UTuple>()))> && ...))
      { // [tuple.assign]-42.4: is_assignable_v<const T_i&, decltype(get<i>(std::forward<UTuple>(u)))> is true for
        // all i
        return (is_nothrow_assignable_v<const _Types&, decltype(get<_Indices>(::cuda::std::declval<_UTuple>()))> && ...)
               ? __select_assignment::__is_nothrow
               : __select_assignment::__may_throw;
      }
      else
      {
        return __select_assignment::__invalid;
      }
    }
    else if constexpr ((is_assignable_v<_Types&, decltype(get<_Indices>(::cuda::std::declval<_UTuple>()))> && ...))
    { // [tuple.assign]-39.4: is_assignable_v<T_i&, decltype(get<i>(std::forward<UTuple>(u)))> is true for all i
      return (is_nothrow_assignable_v<_Types&, decltype(get<_Indices>(::cuda::std::declval<_UTuple>()))> && ...)
             ? __select_assignment::__is_nothrow
             : __select_assignment::__may_throw;
    }
    else
    {
      return __select_assignment::__invalid;
    }
    // NOLINTEND(bugprone-branch-clone)
  }

  template <class... _UTypes>
  static constexpr bool __tuple_all_equality_comparable_v = (__is_cpp17_equality_comparable_v<_Types, _UTypes> && ...);

  template <class... _UTypes>
  static constexpr bool __tuple_all_nothrow_equality_comparable_v =
    (__is_cpp17_nothrow_equality_comparable_v<_Types, _UTypes> && ...);

  template <class... _UTypes>
  static constexpr bool __tuple_all_less_than_comparable_v =
    (__is_cpp17_less_than_comparable_v<_Types, _UTypes> && ...);

  template <class... _UTypes>
  static constexpr bool __tuple_all_nothrow_less_than_comparable_v =
    (__is_cpp17_nothrow_less_than_comparable_v<_Types, _UTypes> && ...);
};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TUPLE_TUPLE_CONSTRAINTS_H
