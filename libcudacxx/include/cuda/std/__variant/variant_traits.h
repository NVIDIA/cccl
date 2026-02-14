//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___VARIANT_VARIANT_TRAITS_H
#define _CUDA_STD___VARIANT_VARIANT_TRAITS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__fwd/variant.h>
#include <cuda/std/__type_traits/add_const.h>
#include <cuda/std/__type_traits/add_cv.h>
#include <cuda/std/__type_traits/add_volatile.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_copy_assignable.h>
#include <cuda/std/__type_traits/is_copy_constructible.h>
#include <cuda/std/__type_traits/is_destructible.h>
#include <cuda/std/__type_traits/is_move_assignable.h>
#include <cuda/std/__type_traits/is_move_constructible.h>
#include <cuda/std/__type_traits/is_trivially_copy_assignable.h>
#include <cuda/std/__type_traits/is_trivially_copy_constructible.h>
#include <cuda/std/__type_traits/is_trivially_destructible.h>
#include <cuda/std/__type_traits/is_trivially_move_assignable.h>
#include <cuda/std/__type_traits/is_trivially_move_constructible.h>
#include <cuda/std/__type_traits/type_list.h>
#include <cuda/std/initializer_list>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT variant_size<const _Tp> : variant_size<_Tp>
{};

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT variant_size<volatile _Tp> : variant_size<_Tp>
{};

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT variant_size<const volatile _Tp> : variant_size<_Tp>
{};

template <class... _Types>
struct _CCCL_TYPE_VISIBILITY_DEFAULT variant_size<variant<_Types...>> : integral_constant<size_t, sizeof...(_Types)>
{};

template <size_t _Ip, class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT variant_alternative<_Ip, const _Tp> : add_const<variant_alternative_t<_Ip, _Tp>>
{};

template <size_t _Ip, class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
variant_alternative<_Ip, volatile _Tp> : add_volatile<variant_alternative_t<_Ip, _Tp>>
{};

template <size_t _Ip, class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
variant_alternative<_Ip, const volatile _Tp> : add_cv<variant_alternative_t<_Ip, _Tp>>
{};

template <size_t _Ip, class... _Types>
struct _CCCL_TYPE_VISIBILITY_DEFAULT variant_alternative<_Ip, variant<_Types...>>
{
  static_assert(_Ip < sizeof...(_Types), "Index out of bounds in cuda::std::variant_alternative<>");
  using type = __type_index_c<_Ip, _Types...>;
};

[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL int __choose_index_type(unsigned int __num_elem) noexcept
{
  constexpr unsigned char __small   = static_cast<unsigned char>(-1);
  constexpr unsigned short __medium = static_cast<unsigned short>(-1);
  if (__num_elem < static_cast<unsigned int>(__small))
  {
    return 0;
  }
  if (__num_elem < static_cast<unsigned int>(__medium))
  {
    return 1;
  }
  return 2;
}

template <size_t _NumAlts>
using __variant_index_t =
  conditional_t<::cuda::std::__choose_index_type(_NumAlts) == 0,
                unsigned char,
                conditional_t<::cuda::std::__choose_index_type(_NumAlts) == 1, unsigned short, unsigned int>>;
namespace __variant_detail
{
struct __valueless_t
{};

enum class _Trait
{
  _TriviallyAvailable,
  _Available,
  _Unavailable
};

template <typename... _Types>
struct __traits
{
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL _Trait __common_trait(initializer_list<_Trait> __traits) noexcept
  {
    _Trait __result = _Trait::_TriviallyAvailable;
    for (_Trait __t : __traits)
    {
      if (static_cast<int>(__t) > static_cast<int>(__result))
      {
        __result = __t;
      }
    }
    return __result;
  }

  template <class _Type>
  static constexpr _Trait __copy_constructible_trait_ =
    is_trivially_copy_constructible_v<_Type> ? _Trait::_TriviallyAvailable
    : is_copy_constructible_v<_Type>
      ? _Trait::_Available
      : _Trait::_Unavailable;

  static constexpr _Trait __copy_constructible_trait = __common_trait({__copy_constructible_trait_<_Types>...});

  template <class _Type>
  static constexpr _Trait __move_constructible_trait_ =
    is_trivially_move_constructible_v<_Type> ? _Trait::_TriviallyAvailable
    : is_move_constructible_v<_Type>
      ? _Trait::_Available
      : _Trait::_Unavailable;

  static constexpr _Trait __move_constructible_trait = __common_trait({__move_constructible_trait_<_Types>...});

  template <class _Type>
  static constexpr _Trait __copy_assignable_trait_ =
    is_trivially_copy_assignable_v<_Type> ? _Trait::_TriviallyAvailable
    : is_copy_assignable_v<_Type>
      ? _Trait::_Available
      : _Trait::_Unavailable;

  static constexpr _Trait __copy_assignable_trait =
    __common_trait({__copy_constructible_trait, __copy_assignable_trait_<_Types>...});

  template <class _Type>
  static constexpr _Trait __move_assignable_trait_ =
    is_trivially_move_assignable_v<_Type> ? _Trait::_TriviallyAvailable
    : is_move_assignable_v<_Type>
      ? _Trait::_Available
      : _Trait::_Unavailable;

  static constexpr _Trait __move_assignable_trait =
    __common_trait({__move_constructible_trait, __move_assignable_trait_<_Types>...});

  template <class _Type>
  static constexpr _Trait __destructible_trait_ =
    is_trivially_destructible_v<_Type> ? _Trait::_TriviallyAvailable
    : is_destructible_v<_Type>
      ? _Trait::_Available
      : _Trait::_Unavailable;

  static constexpr _Trait __destructible_trait = __common_trait({__destructible_trait_<_Types>...});
};
} // namespace __variant_detail

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___VARIANT_VARIANT_TRAITS_H
