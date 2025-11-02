//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___VARIANT_VARIANT_H
#define _CUDA_STD___VARIANT_VARIANT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__fwd/variant.h>
#include <cuda/std/__tuple_dir/sfinae_helpers.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/dependent_type.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_array.h>
#include <cuda/std/__type_traits/is_copy_assignable.h>
#include <cuda/std/__type_traits/is_copy_constructible.h>
#include <cuda/std/__type_traits/is_default_constructible.h>
#include <cuda/std/__type_traits/is_move_assignable.h>
#include <cuda/std/__type_traits/is_move_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_default_constructible.h>
#include <cuda/std/__type_traits/is_reference.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/in_place.h>
#include <cuda/std/__variant/sfinae_helpers.h>
#include <cuda/std/__variant/variant_access.h>
#include <cuda/std/__variant/variant_base.h>
#include <cuda/std/__variant/variant_constraints.h>
#include <cuda/std/__variant/variant_visit.h>
#include <cuda/std/initializer_list>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class... _Types>
class _CCCL_TYPE_VISIBILITY_DEFAULT variant
    : private __variant_base<__all<is_copy_constructible_v<_Types>...>::value,
                             __all<is_move_constructible_v<_Types>...>::value,
                             __all<(is_copy_constructible_v<_Types> && is_copy_assignable_v<_Types>) ...>::value,
                             __all<(is_move_constructible_v<_Types> && is_move_assignable_v<_Types>) ...>::value>
{
  static_assert(0 < sizeof...(_Types), "variant must consist of at least one alternative.");

  static_assert((!is_array_v<_Types> && ...), "variant can not have an array type as an alternative.");

  static_assert((!is_reference_v<_Types> && ...), "variant can not have a reference type as an alternative.");

  static_assert((!is_void_v<_Types> && ...), "variant can not have a void type as an alternative.");

  using __first_type  = variant_alternative_t<0, variant>;
  using __constraints = __variant_detail::__variant_constraints<_Types...>;

public:
  // Needs to be dependent to guard against incomplete types
  template <bool _Dummy = true,
            class       = enable_if_t<__dependent_type<is_default_constructible<__first_type>, _Dummy>::value>>
  _CCCL_API constexpr variant() noexcept(is_nothrow_default_constructible_v<__first_type>)
      : __impl_(in_place_index<0>)
  {}

  _CCCL_HIDE_FROM_ABI constexpr variant(const variant&) = default;
  _CCCL_HIDE_FROM_ABI constexpr variant(variant&&)      = default;

  template <class _Arg>
  using __match_construct =
    conditional_t<!is_same_v<remove_cvref_t<_Arg>, variant> && !__is_cuda_std_inplace_type_v<remove_cvref_t<_Arg>> //
                    && !__is_cuda_std_inplace_index_v<remove_cvref_t<_Arg>>,
                  typename __constraints::template __match_construct<_Arg>,
                  __variant_detail::__invalid_variant_constraints>;

  // CTAD fails if we do not SFINAE the empty variant away first
  template <class _Arg,
            class              = enable_if_t<sizeof...(_Types) != 0>,
            class _Constraints = __match_construct<_Arg>,
            class              = enable_if_t<_Constraints::__constructible>>
  _CCCL_API constexpr variant(_Arg&& __arg) noexcept(_Constraints::__nothrow_constructible)
      : __impl_(in_place_index<_Constraints::_Ip>, ::cuda::std::forward<_Arg>(__arg))
  {}

  template <size_t _Ip, class... _Args>
  using __variadic_construct =
    conditional_t<(_Ip < sizeof...(_Types)),
                  typename __constraints::template __variadic_construct<_Ip, _Args...>,
                  __variant_detail::__invalid_variant_constraints>;

  template <size_t _Ip,
            class... _Args,
            class _Constraints = __variadic_construct<_Ip, _Args...>,
            class              = enable_if_t<_Constraints::__constructible>>
  _CCCL_API explicit constexpr variant(in_place_index_t<_Ip>,
                                       _Args&&... __args) noexcept(_Constraints::__nothrow_constructible)
      : __impl_(in_place_index<_Ip>, ::cuda::std::forward<_Args>(__args)...)
  {}

  template <size_t _Ip, class _Up, class... _Args>
  using __variadic_ilist_construct =
    conditional_t<(_Ip < sizeof...(_Types)),
                  typename __constraints::template __variadic_ilist_construct<_Ip, _Up, _Args...>,
                  __variant_detail::__invalid_variant_constraints>;

  template <size_t _Ip,
            class _Up,
            class... _Args,
            class _Constraints = __variadic_ilist_construct<_Ip, _Up, _Args...>,
            class              = enable_if_t<_Constraints::__constructible>>
  _CCCL_API explicit constexpr variant(in_place_index_t<_Ip>, initializer_list<_Up> __il, _Args&&... __args) noexcept(
    _Constraints::__nothrow_constructible)
      : __impl_(in_place_index<_Ip>, __il, ::cuda::std::forward<_Args>(__args)...)
  {}

  template <class _Tp,
            class... _Args,
            size_t _Ip         = __find_detail::__find_unambiguous_index_sfinae<_Tp, _Types...>::value,
            class _Constraints = __variadic_construct<_Ip, _Args...>,
            class              = enable_if_t<_Constraints::__constructible>>
  _CCCL_API explicit constexpr variant(in_place_type_t<_Tp>,
                                       _Args&&... __args) noexcept(_Constraints::__nothrow_constructible)
      : __impl_(in_place_index<_Ip>, ::cuda::std::forward<_Args>(__args)...)
  {}

  template <class _Tp,
            class _Up,
            class... _Args,
            size_t _Ip         = __find_detail::__find_unambiguous_index_sfinae<_Tp, _Types...>::value,
            class _Constraints = __variadic_ilist_construct<_Ip, _Up, _Args...>,
            class              = enable_if_t<_Constraints::__constructible>>
  _CCCL_API explicit constexpr variant(in_place_type_t<_Tp>, initializer_list<_Up> __il, _Args&&... __args) noexcept(
    _Constraints::__nothrow_constructible)
      : __impl_(in_place_index<_Ip>, __il, ::cuda::std::forward<_Args>(__args)...)
  {}

  _CCCL_HIDE_FROM_ABI ~variant() = default;

  _CCCL_HIDE_FROM_ABI constexpr variant& operator=(const variant&) = default;
  _CCCL_HIDE_FROM_ABI constexpr variant& operator=(variant&&)      = default;

  template <class _Arg>
  using __match_assign =
    conditional_t<!is_same_v<remove_cvref_t<_Arg>, variant>,
                  typename __constraints::template __match_assign<_Arg>,
                  __variant_detail::__invalid_variant_constraints>;

  template <class _Arg, class _Constraints = __match_assign<_Arg>, class = enable_if_t<_Constraints::__assignable>>
  _CCCL_API inline variant& operator=(_Arg&& __arg) noexcept(_Constraints::__nothrow_assignable)
  {
    __impl_.template __assign<_Constraints::_Ip>(::cuda::std::forward<_Arg>(__arg));
    return *this;
  }

  template <size_t _Ip,
            class... _Args,
            class _Constraints = __variadic_construct<_Ip, _Args...>,
            class              = enable_if_t<_Constraints::__constructible>>
  _CCCL_API inline typename _Constraints::_Tp& emplace(_Args&&... __args)
  {
    return __impl_.template __emplace<_Ip>(::cuda::std::forward<_Args>(__args)...);
  }

  template <size_t _Ip,
            class _Up,
            class... _Args,
            class _Constraints = __variadic_ilist_construct<_Ip, _Up, _Args...>,
            class              = enable_if_t<_Constraints::__constructible>>
  _CCCL_API inline typename _Constraints::_Tp& emplace(initializer_list<_Up> __il, _Args&&... __args)
  {
    return __impl_.template __emplace<_Ip>(__il, ::cuda::std::forward<_Args>(__args)...);
  }

  template <class _Tp,
            class... _Args,
            size_t _Ip = __find_detail::__find_unambiguous_index_sfinae<_Tp, _Types...>::value,
            enable_if_t<is_constructible_v<_Tp, _Args...>, int> = 0>
  _CCCL_API inline _Tp& emplace(_Args&&... __args)
  {
    return __impl_.template __emplace<_Ip>(::cuda::std::forward<_Args>(__args)...);
  }

  template <class _Tp,
            class _Up,
            class... _Args,
            size_t _Ip = __find_detail::__find_unambiguous_index_sfinae<_Tp, _Types...>::value,
            enable_if_t<is_constructible_v<_Tp, initializer_list<_Up>&, _Args...>, int> = 0>
  _CCCL_API inline _Tp& emplace(initializer_list<_Up> __il, _Args&&... __args)
  {
    return __impl_.template __emplace<_Ip>(__il, ::cuda::std::forward<_Args>(__args)...);
  }

  [[nodiscard]] _CCCL_API constexpr bool valueless_by_exception() const noexcept
  {
    return __impl_.valueless_by_exception();
  }

  [[nodiscard]] _CCCL_API constexpr size_t index() const noexcept
  {
    return __impl_.index();
  }

  // Needs to be dependent to guard against incomplete types
  template <bool _Dummy>
  using __swap_constraint =
    __dependent_type<typename __variant_detail::__variant_constraints<_Types...>::template __swappable<_Dummy>, _Dummy>;

  template <bool _Dummy       = true,
            class _Constraint = __swap_constraint<_Dummy>,
            class             = enable_if_t<_Constraint::__is_swappable_v>>
  _CCCL_API inline void swap(variant& __that) noexcept(_Constraint::__is_nothrow_swappable_v)
  {
    __impl_.__swap(__that.__impl_);
  }

  _CCCL_API static constexpr size_t __size() noexcept
  {
    return sizeof...(_Types);
  }

private:
  __variant_detail::__impl<_Types...> __impl_;

  friend struct __variant_detail::__access::__variant;
  friend struct __variant_detail::__visitation::__variant;
};

template <class... _Types>
_CCCL_API inline auto swap(variant<_Types...>& __lhs, variant<_Types...>& __rhs) noexcept(noexcept(__lhs.swap(__rhs)))
  -> decltype(__lhs.swap(__rhs))
{
  return __lhs.swap(__rhs);
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___VARIANT_VARIANT_H
