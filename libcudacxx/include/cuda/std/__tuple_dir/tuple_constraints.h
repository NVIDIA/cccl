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

// __tuple_like_with_size
template <class _Tuple, size_t _ExpectedSize, bool = __tuple_like<_Tuple>>
inline constexpr bool __tuple_like_with_size = false;

template <class _Tuple, size_t _ExpectedSize>
inline constexpr bool __tuple_like_with_size<_Tuple, _ExpectedSize, true> =
  _ExpectedSize == tuple_size<remove_cvref_t<_Tuple>>::value;

template <class... _Types>
struct __tuple_constraints;

template <class... _Types>
[[nodiscard]] _CCCL_TRIVIAL_API _CCCL_CONSTEVAL auto __tuple_get_constraints(__tuple_types<_Types...>) noexcept
{
  return __tuple_constraints<_Types...>{};
}

template <class... _Types>
struct __tuple_constraints
{
  template <int = 0>
  [[nodiscard]] _CCCL_TRIVIAL_API static _CCCL_CONSTEVAL __select_constructor __select_default_constructible() noexcept
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
  [[nodiscard]] _CCCL_TRIVIAL_API static _CCCL_CONSTEVAL __select_constructor
  __select_variadic_copy_constructible() noexcept
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
  [[nodiscard]] _CCCL_TRIVIAL_API static _CCCL_CONSTEVAL __select_constructor
  __select_variadic_move_constructible() noexcept
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
  [[nodiscard]] _CCCL_TRIVIAL_API static _CCCL_CONSTEVAL bool __disambiguate_variadic_constructible() noexcept
  {
    // NOLINTBEGIN(bugprone-branch-clone)
    if constexpr (sizeof...(_Types) == 0)
    {
      return false;
    }
    else if constexpr (sizeof...(_Types) != sizeof...(_UTypes))
    {
      return false;
    }
    else if constexpr (sizeof...(_Types) == 1)
    { // [tuple.cnstr]-12.1: negation<is_same<remove_cvref_t<U0>, tuple>> if sizeof...(Types) is 1
      using _U0 = __type_index_c<0, _UTypes...>;
      using _T0 = __type_index_c<0, _Types...>;
      return !is_same_v<remove_cvref_t<_U0>, tuple<_T0>>;
    }
    else if constexpr (sizeof...(_Types) == 2 || sizeof...(_Types) == 3)
    { // [tuple.cnstr]-12.2: otherwise, if sizeof...(Types) is 2 or 3
      // [tuple.cnstr]-13.3: !is_same_v<remove_cvref_t<U0>, allocator_arg_t> || is_same_v<remove_cvref_t<T0>,
      // allocator_arg_t>>
      using _U0 = __type_index_c<0, _UTypes...>;
      using _T0 = __type_index_c<0, _Types...>;
      return !is_same_v<remove_cvref_t<_U0>, allocator_arg_t> || is_same_v<remove_cvref_t<_T0>, allocator_arg_t>;
    }
    else
    {
      return true;
    }
    // NOLINTEND(bugprone-branch-clone)
  }

  template <class _UTuple>
  [[nodiscard]] _CCCL_TRIVIAL_API static _CCCL_CONSTEVAL bool __disambiguate_tuple_like() noexcept
  {
    // NOLINTBEGIN(bugprone-branch-clone)
    if constexpr (sizeof...(_Types) == 0)
    {
      return false;
    }
    else if constexpr (__is_cuda_std_ranges_subrange_v<remove_cvref_t<_UTuple>>)
    { // [tuple#cnstr]-29.2: remove_cvref_t<UTuple> is not a specialization of ranges::subrange,
      return false;
    }
    else if constexpr (is_same_v<_UTuple, const tuple<_Types...>&> || is_same_v<_UTuple, tuple<_Types...>&&>)
    { // Prefers the copy/move constructor
      return false;
    }
    else if constexpr (!__tuple_like_with_size<_UTuple, sizeof...(_Types)>)
    { // [tuple#cnstr]-21.1: sizeof...(Types) equals sizeof...(UTypes), and
      // [tuple#cnstr]-25.1: sizeof...(Types) is 2, (pair constructor)
      // [tuple#cnstr]-29.3: sizeof...(Types) equals sizeof...(UTypes), and
      return false;
    }
    else if constexpr (sizeof...(_Types) == 1)
    {
      using _U0 = tuple_element_t<0, remove_cvref_t<_UTuple>>;
      using _T0 = __type_index_c<0, _Types...>;
      if constexpr (__is_cuda_std_tuple<remove_cvref_t<_UTuple>> && is_same_v<_T0, _U0>)
      { // [tuple#cnstr]-21.3: either sizeof...(Types) is not 1
        // [tuple#cnstr]-21.3: is_same_v<T, U> is false
        return false;
      }
      else if constexpr (is_constructible_v<_T0, _UTuple>)
      { // [tuple#cnstr]-21.3: either sizeof...(Types) is not 1, or is_constructible_v<T, _UTuple> are false
        // [tuple#cnstr]-29.5: either sizeof...(Types) is not 1, or is_constructible_v<T, _UTuple> are false
        return false;
      }
      else if constexpr (is_convertible_v<_UTuple, _T0>)
      { // [tuple#cnstr]-21.3: either sizeof...(Types) is not 1, or is_convertible_v<_UTuple, T> are false
        // [tuple#cnstr]-29.5: either sizeof...(Types) is not 1, or is_convertible_v<_UTuple, T> are false
        return false;
      }
      else
      {
        return true;
      }
    }
    else
    {
      return true;
    }
    // NOLINTEND(bugprone-branch-clone)
  }

#if _CCCL_COMPILER(MSVC)
  // MSVC crashes when the disambiguation functions are called directly inside an enable_if while synthesizing
  // the implicit deduction guides, so go through a variable template wrapped into a bool_constant
  template <class... _UTypes>
  static constexpr bool __disambiguate_variadic_v = __disambiguate_variadic_constructible<_UTypes...>();

  template <class _UTuple>
  static constexpr bool __disambiguate_tuple_like_v = __disambiguate_tuple_like<_UTuple>();
#endif // _CCCL_COMPILER(MSVC)

  template <class... _UTypes>
  [[nodiscard]] _CCCL_TRIVIAL_API static _CCCL_CONSTEVAL __select_constructor __select_variadic_constructible() noexcept
  {
    // NOLINTBEGIN(bugprone-branch-clone)
    if constexpr (sizeof...(_Types) != sizeof...(_UTypes))
    {
      return __select_constructor::__invalid;
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

  template <class... _UTypes>
  [[nodiscard]] _CCCL_TRIVIAL_API static _CCCL_CONSTEVAL __select_constructor
  __select_variadic_constructible_less_rank() noexcept
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
    { // MSVC has issues with constexpr variables here, so no constexpr variable
      using __arg_list        = __make_tuple_types_t<__tuple_types<_Types...>, sizeof...(_UTypes)>;
      using __arg_constraints = decltype(::cuda::std::__tuple_get_constraints(__arg_list{}));
      if constexpr (!__arg_constraints::template __disambiguate_variadic_constructible<_UTypes...>())
      {
        return __select_constructor::__invalid;
      }
      else if constexpr (__arg_constraints::template __select_variadic_constructible<_UTypes...>()
                           == __select_constructor::__invalid
                         || __arg_constraints::template __select_variadic_constructible<_UTypes...>()
                              == __select_constructor::__deleted)
      {
        return __select_constructor::__invalid;
      }
      else
      {
        using __defaulted_list = __make_tuple_types_t<__tuple_types<_Types...>, sizeof...(_Types), sizeof...(_UTypes)>;
        using __defaulted_constraints = decltype(::cuda::std::__tuple_get_constraints(__defaulted_list{}));
        if constexpr (__defaulted_constraints::__select_default_constructible() == __select_constructor::__implicit
                      || __defaulted_constraints::__select_default_constructible() == __select_constructor::__explicit)
        {
          return __select_constructor::__explicit;
        }
        else
        {
          return __select_constructor::__invalid;
        }
      }
    }
    // NOLINTEND(bugprone-branch-clone)
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <class _UTuple, size_t... _Indices>
  [[nodiscard]] _CCCL_TRIVIAL_API static _CCCL_CONSTEVAL __select_constructor
  __select_tuple_like_constructible(__tuple_indices<_Indices...>) noexcept
  {
    using ::cuda::std::get;
    return __select_variadic_constructible<decltype(get<_Indices>(::cuda::std::declval<_UTuple>()))...>();
  }

  template <class _UTuple>
  static constexpr __select_constructor __select_tuple_like_constructible_v =
    __select_tuple_like_constructible<_UTuple>(__make_tuple_indices_t<sizeof...(_Types)>{});

  _CCCL_EXEC_CHECK_DISABLE
  template <class _UTuple, size_t... _Indices>
  [[nodiscard]] _CCCL_TRIVIAL_API static _CCCL_CONSTEVAL bool
  __nothrow_tuple_like_constructible(__tuple_indices<_Indices...>) noexcept
  {
    using ::cuda::std::get;
    return (is_nothrow_constructible_v<_Types, decltype(get<_Indices>(::cuda::std::declval<_UTuple>()))> && ...);
  }

  template <class _UTuple>
  static constexpr bool __nothrow_tuple_like_constructible_v =
    __nothrow_tuple_like_constructible<_UTuple>(__make_tuple_indices_t<sizeof...(_Types)>{});

  // Assignments
  static constexpr bool __all_copy_assignable = (is_copy_assignable_v<_Types> && ...);
  static constexpr bool __all_move_assignable = (is_move_assignable_v<_Types> && ...);

  // [tuple.assign]-5: is_copy_assignable_v<const Types> is true for all i.
  static constexpr bool __all_const_copy_assignable = (is_copy_assignable_v<const _Types> && ...);
  // [tuple.assign]-12: is_assignable_v<const Types&, Types> is true for all i.
  static constexpr bool __all_const_move_assignable = (is_assignable_v<const _Types&, _Types> && ...);

  template <bool _IsConst, class... _UTypes>
  [[nodiscard]] _CCCL_TRIVIAL_API static _CCCL_CONSTEVAL bool __select_converting_assignable() noexcept
  {
    // NOLINTBEGIN(bugprone-branch-clone)
    if constexpr (sizeof...(_Types) != sizeof...(_UTypes))
    { // [tuple.assign]-15.1: sizeof...(Types) equals sizeof...(UTypes) and
      return false;
    }
    else if constexpr (is_same_v<tuple<_Types...>, tuple<_UTypes...>>)
    { // Disambiguate the non-converting assignments
      return false;
    }
    else if constexpr (_IsConst)
    { // [tuple.assign]-18.2: is_assignable_v<const Types&, const UTypes&> is true for all i.
      // [tuple.assign]-24.2: is_assignable_v<const Types&, UTypes> is true for all i.
      return (is_assignable_v<const _Types&, _UTypes> && ...);
    }
    else
    { // [tuple.assign]-15.2: is_assignable_v<Types&, const UTypes&> is true for all i.
      // [tuple.assign]-21.2: is_assignable_v<Types&, UTypes> is true for all i.
      return (is_assignable_v<_Types&, _UTypes> && ...);
    }
    // NOLINTEND(bugprone-branch-clone)
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <bool _IsConst, class _UTuple, size_t... _Indices>
  [[nodiscard]] _CCCL_TRIVIAL_API static _CCCL_CONSTEVAL bool
  __select_tuple_like_assignable(__tuple_indices<_Indices...>) noexcept
  {
    using ::cuda::std::get;
    // NOLINTBEGIN(bugprone-branch-clone)
    if constexpr (is_same_v<remove_cvref_t<_UTuple>, tuple<_Types...>>)
    { // [tuple.assign]-39.1: different-from<UTuple, tuple>
      return false;
    }
    else if constexpr (__is_cuda_std_ranges_subrange_v<remove_cvref_t<_UTuple>>)
    { // [tuple.assign]-39.2: remove_cvref_t<UTuple> is not a specialization of ranges::subrange,
      return false;
    }
    else if constexpr (!__tuple_like_with_size<_UTuple, sizeof...(_Types)>)
    { // [tuple.assign]-39.3: sizeof...(Types) equals tuple_size_v<remove_cvref_t<UTuple>>, and
      return false;
    }
    else if constexpr (_IsConst)
    { // [tuple.assign]-42.4: is_assignable_v<const T_i&, decltype(get<i>(std::forward<UTuple>(u)))> is true for
      // all i
      return (is_assignable_v<const _Types&, decltype(get<_Indices>(::cuda::std::declval<_UTuple>()))> && ...);
    }
    else
    { // [tuple.assign]-39.4: is_assignable_v<T_i&, decltype(get<i>(std::forward<UTuple>(u)))> is true for all i
      return (is_assignable_v<_Types&, decltype(get<_Indices>(::cuda::std::declval<_UTuple>()))> && ...);
    }
    // NOLINTEND(bugprone-branch-clone)
  }
  _CCCL_EXEC_CHECK_DISABLE
  template <bool _IsConst, class _UTuple>
  [[nodiscard]] _CCCL_TRIVIAL_API static _CCCL_CONSTEVAL bool __select_tuple_like_assignable() noexcept
  {
    return __select_tuple_like_assignable<_IsConst, _UTuple>(__make_tuple_indices_t<sizeof...(_Types)>{});
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <bool _IsConst, class _UTuple, size_t... _Indices>
  [[nodiscard]] _CCCL_TRIVIAL_API static _CCCL_CONSTEVAL bool
  __nothrow_tuple_like_assignable(__tuple_indices<_Indices...>) noexcept
  {
    using ::cuda::std::get;
    if constexpr (_IsConst)
    {
      return (is_nothrow_assignable_v<const _Types&, decltype(get<_Indices>(::cuda::std::declval<_UTuple>()))> && ...);
    }
    else
    {
      return (is_nothrow_assignable_v<_Types&, decltype(get<_Indices>(::cuda::std::declval<_UTuple>()))> && ...);
    }
  }
  _CCCL_EXEC_CHECK_DISABLE
  template <bool _IsConst, class _UTuple>
  [[nodiscard]] _CCCL_TRIVIAL_API static _CCCL_CONSTEVAL bool __nothrow_tuple_like_assignable() noexcept
  {
    return __nothrow_tuple_like_assignable<_IsConst, _UTuple>(__make_tuple_indices_t<sizeof...(_Types)>{});
  }

  // Comparisons
  template <class... _UTypes>
  [[nodiscard]] _CCCL_TRIVIAL_API static _CCCL_CONSTEVAL bool __is_equality_comparable() noexcept
  {
    if constexpr (sizeof...(_Types) != sizeof...(_UTypes))
    {
      return false;
    }
    else
    {
      return (__is_cpp17_equality_comparable_v<_Types, _UTypes> && ...);
    }
  }

  template <class... _UTypes>
  static constexpr bool __is_equality_comparable_v = __is_equality_comparable<_UTypes...>();

  template <class... _UTypes>
  static constexpr bool __is_nothrow_equality_comparable_v =
    (__is_cpp17_nothrow_equality_comparable_v<_Types, _UTypes> && ...);

  template <class... _UTypes>
  [[nodiscard]] _CCCL_TRIVIAL_API static _CCCL_CONSTEVAL bool __is_less_than_comparable() noexcept
  {
    if constexpr (sizeof...(_Types) != sizeof...(_UTypes))
    {
      return false;
    }
    else
    {
      return (__is_cpp17_less_than_comparable_v<_Types, _UTypes> && ...);
    }
  }

  template <class... _UTypes>
  static constexpr bool __is_less_than_comparable_v = __is_less_than_comparable<_UTypes...>();

  template <class... _UTypes>
  static constexpr bool __is_nothrow_less_than_comparable_v =
    (__is_cpp17_nothrow_less_than_comparable_v<_Types, _UTypes> && ...);
};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TUPLE_TUPLE_CONSTRAINTS_H
