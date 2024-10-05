//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA__MDSPAN_ACCESSOR_WITH_PROPERTIES_H
#define _CUDA__MDSPAN_ACCESSOR_WITH_PROPERTIES_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_STD_VER >= 2014

// #  include <cub/thread/thread_load.cuh>
// #  include <cub/thread/thread_store.cuh>

#  include <cuda/annotated_ptr>
#  include <cuda/std/__concepts/__concept_macros.h>
#  include <cuda/std/__type_traits/enable_if.h>
#  include <cuda/std/__type_traits/is_abstract.h>
#  include <cuda/std/__type_traits/is_array.h>
#  include <cuda/std/__type_traits/is_convertible.h>
#  include <cuda/std/__type_traits/type_list.h>

#  include "cuda/std/__type_traits/integral_constant.h"
#  include "cuda/std/__type_traits/is_same.h"

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

// #  if 0
// template <class _ElementType,
//           eviction_policy _Eviction = eviction_policy::Default,
//           prefetch_size _Prefetch   = prefetch_size::no_prefetch,
//           typename _CacheHint      = access_property::normal,
//           typename _Enable         = void>
// struct accessor_with_properties;
//
///***********************************************************************************************************************
// * accessor_reference
// **********************************************************************************************************************/
//
// template <class _ElementType, eviction_policy _Eviction, prefetch_size _Prefetch, typename _CacheHint>
// class accessor_reference
//{
//  using __pointer_type = _ElementType*;
//
//  __pointer_type __p;
//
//  friend class accessor_with_properties<_ElementType, _Eviction, _Prefetch, _CacheHint>;
//
// public:
//  explicit constexpr accessor_reference() noexcept = default;
//
//  accessor_reference(accessor_reference&&) = delete;
//
//  accessor_reference& operator=(accessor_reference&&) = delete;
//
//  _CCCL_HIDE_FROM_ABI _CCCL_DEVICE constexpr accessor_reference(const accessor_reference&) = default;
//
//  _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _CCCL_FORCEINLINE accessor_reference&
//  operator=(const accessor_reference& __x) noexcept
//  {
//    return operator=(static_cast<_ElementType>(__x));
//  }
//
//  _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _CCCL_FORCEINLINE accessor_reference& operator=(_ElementType __x) noexcept
//  {
//    return cub::ThreadStore<_Eviction>(__p, __x);
//  }
//
//  _CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE operator _ElementType() const noexcept
//  {
//    return cub::ThreadLoad<_Eviction>(__p);
//  }
//
// private:
//  _CCCL_HIDE_FROM_ABI _CCCL_DEVICE explicit accessor_reference(__pointer_type __p_) noexcept
//      : __p{__p_}
//  {}
//};
//
///***********************************************************************************************************************
// * load/store accessor_with_properties
// **********************************************************************************************************************/
//
// template <class _ElementType, eviction_policy _Eviction, prefetch_size _Prefetch, typename _CacheHint>
// struct accessor_with_properties<_ElementType,
//                             _Eviction,
//                             _Prefetch,
//                             _CacheHint,
//                             _CUDA_VSTD::__enable_if_t<!_CUDA_VSTD::is_const<_ElementType>::value>>
//{
//  static_assert(!_CUDA_VSTD::is_array<_ElementType>::value,
//                "accessor_with_properties: template argument may not be an array type");
//  static_assert(!_CUDA_VSTD::is_abstract<_ElementType>::value,
//                "accessor_with_properties: template argument may not be an abstract class");
//
//  using offset_policy = accessor_with_properties;
//  using element_type  = _ElementType;
//  using reference     = accessor_reference<_ElementType, _Eviction, _Prefetch, _CacheHint>;
//  using data_handle_type =
//    typename _CUDA_VSTD::conditional<RestrictProperty, _ElementType*, _ElementType * _CCCL_RESTRICT>::type;
//
//  explicit constexpr accessor_with_properties() noexcept = default;
//
//  _LIBCUDACXX_TEMPLATE(class _OtherElementType)
//  _LIBCUDACXX_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_convertible, _OtherElementType (*)[], element_type (*)[]))
//  _CCCL_HIDE_FROM_ABI _CCCL_DEVICE constexpr accessor_with_properties(accessor_with_properties<_OtherElementType>)
//  noexcept {}
//
//  _CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE reference access(data_handle_type __p, size_t __i) const noexcept
//  {
//    return reference{__p + __i};
//  }
//
//  _CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE data_handle_type offset(data_handle_type __p, size_t __i) const
//  noexcept
//  {
//    return __p + __i;
//  }
//};
// #  endif
/***********************************************************************************************************************
 * load-only accessor_with_properties
 **********************************************************************************************************************/

#  define _CCCL_RESTRICT

template <typename _ElementType, typename... P>
class accessor_reference
{};

/***********************************************************************************************************************
 * Eviction Policies
 **********************************************************************************************************************/

enum class _EvictionPolicyEnum
{
  _Default,
  _First,
  _Normal,
  _Last,
  _LastUse,
  _NoAllocation,
};

template <_EvictionPolicyEnum _Value>
using __eviction_policy_constant_t = _CUDA_VSTD::integral_constant<_EvictionPolicyEnum, _Value>;

template <_EvictionPolicyEnum _Value>
constexpr auto __eviction_policy_constant = __eviction_policy_constant_t<_Value>{};

template <typename>
struct __is_eviction_policy : _CUDA_VSTD::false_type
{};

template <_EvictionPolicyEnum _Value>
struct __is_eviction_policy<__eviction_policy_constant_t<_Value>> : _CUDA_VSTD::true_type
{};

namespace eviction_policy
{

constexpr auto Default       = __eviction_policy_constant<_EvictionPolicyEnum::_Default>;
constexpr auto first         = __eviction_policy_constant<_EvictionPolicyEnum::_First>;
constexpr auto normal        = __eviction_policy_constant<_EvictionPolicyEnum::_Normal>;
constexpr auto last          = __eviction_policy_constant<_EvictionPolicyEnum::_Last>;
constexpr auto last_use      = __eviction_policy_constant<_EvictionPolicyEnum::_LastUse>;
constexpr auto no_allocation = __eviction_policy_constant<_EvictionPolicyEnum::_NoAllocation>;

}; // namespace eviction_policy

/***********************************************************************************************************************
 * Prefetch Size
 **********************************************************************************************************************/

enum class _PrefetchSizeEnum
{
  _NoPrefetch,
  _Bytes64,
  _Bytes128,
  _Bytes256,
};

template <_PrefetchSizeEnum _Value>
constexpr auto __prefetch_constant = _CUDA_VSTD::integral_constant<_PrefetchSizeEnum, _Value>{};

template <typename>
struct __is_prefetch_policy : _CUDA_VSTD::false_type
{};

template <_PrefetchSizeEnum _Value>
struct __is_prefetch_policy<_CUDA_VSTD::integral_constant<_PrefetchSizeEnum, _Value>> : _CUDA_VSTD::true_type
{};

namespace prefetch_size
{

constexpr auto no_prefetch = __prefetch_constant<_PrefetchSizeEnum::_NoPrefetch>;
constexpr auto bytes_64    = __prefetch_constant<_PrefetchSizeEnum::_Bytes64>;
constexpr auto bytes_128   = __prefetch_constant<_PrefetchSizeEnum::_Bytes128>;
constexpr auto bytes_256   = __prefetch_constant<_PrefetchSizeEnum::_Bytes256>;

}; // namespace prefetch_size

/***********************************************************************************************************************
 * Aliasing Policies
 **********************************************************************************************************************/

enum class _AliasingPolicyEnum
{
  _Restrict,
  _MayAlias
};

template <_AliasingPolicyEnum _Value>
constexpr auto __aliasing_constant = _CUDA_VSTD::integral_constant<_AliasingPolicyEnum, _Value>{};

template <typename>
struct __is_aliasing_policy : _CUDA_VSTD::false_type
{};

template <_AliasingPolicyEnum _Value>
struct __is_aliasing_policy<_CUDA_VSTD::integral_constant<_AliasingPolicyEnum, _Value>> : _CUDA_VSTD::true_type
{};

namespace aliasing_policy
{

constexpr auto restrict  = __aliasing_constant<_AliasingPolicyEnum::_Restrict>;
constexpr auto may_alias = __aliasing_constant<_AliasingPolicyEnum::_MayAlias>;

}; // namespace aliasing_policy

/***********************************************************************************************************************
 * Alignment
 **********************************************************************************************************************/

template <typename>
struct __is_alignment : _CUDA_VSTD::false_type
{};

template <size_t __AlignBytes>
struct __is_alignment<aligned_size_t<__AlignBytes>> : _CUDA_VSTD::true_type
{};

/***********************************************************************************************************************
 * Cache Hints
 **********************************************************************************************************************/

template <typename T>
struct __is_cache_hint
    : _CUDA_VSTD::bool_constant<_CUDA_VSTD::is_same<T, access_property::streaming>::value
                                || _CUDA_VSTD::is_same<T, access_property::persisting>::value
                                || _CUDA_VSTD::is_same<T, access_property::normal>::value
                                || _CUDA_VSTD::is_same<T, access_property::global>::value
                                || _CUDA_VSTD::is_same<T, access_property::shared>::value>
{};

/***********************************************************************************************************************
 * Find Duplicate Utilities
 **********************************************************************************************************************/

template <template <typename> class _Predicate>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_count_if
{
  template <class... _Ts>
  using __call _LIBCUDACXX_NODEBUG_TYPE = _CUDA_VSTD::integral_constant<int, (_Predicate<_Ts>::value + ...)>;
};

template <class... _Ts>
using __type_count_eviction = _CUDA_VSTD::__type_call<__type_count_if<__is_eviction_policy>, _Ts...>;

template <class... _Ts>
using __type_count_prefetch = _CUDA_VSTD::__type_call<__type_count_if<__is_prefetch_policy>, _Ts...>;

template <class... _Ts>
using __type_count_alignment = _CUDA_VSTD::__type_call<__type_count_if<__is_alignment>, _Ts...>;

template <class... _Ts>
using __type_count_aliasing = _CUDA_VSTD::__type_call<__type_count_if<__is_aliasing_policy>, _Ts...>;

template <class... _Ts>
using __type_count_cache_hint = _CUDA_VSTD::__type_call<__type_count_if<__is_cache_hint>, _Ts...>;

/***********************************************************************************************************************
 * Find Properties
 **********************************************************************************************************************/

template <template <typename> class _Predicate>
struct __predicate_call
{
  template <typename _Property>
  using __call = _CUDA_VSTD::bool_constant<_Predicate<_Property>::value>;
};

template <template <typename> class _Predicate, typename DefaultValue, typename... _UserProperties>
struct __find_property
{
  using __ret = _CUDA_VSTD::__type_find_if<_CUDA_VSTD::__type_list<_UserProperties...>, __predicate_call<_Predicate>>;

  using __type = _CUDA_VSTD::__type_front<_CUDA_VSTD::__type_concat<__ret, _CUDA_VSTD::__type_list<DefaultValue>>>;
};

template <typename... _UserProperties>
using __find_eviction_policy =
  __find_property<__is_eviction_policy, decltype(eviction_policy::Default), _UserProperties...>;

template <typename... _UserProperties>
using __find_prefetch_size =
  __find_property<__is_prefetch_policy, decltype(prefetch_size::no_prefetch), _UserProperties...>;

template <_CUDA_VSTD::size_t DefaultAlignment, typename... _UserProperties>
using __find_aligment = __find_property<__is_alignment, aligned_size_t<DefaultAlignment>, _UserProperties...>;

template <typename... _UserProperties>
using __find_aliasing_policy =
  __find_property<__is_aliasing_policy, decltype(aliasing_policy::restrict), _UserProperties...>;

template <typename... _UserProperties>
using __find_cache_hint_policy = __find_property<__is_cache_hint, access_property::global, _UserProperties...>;

/***********************************************************************************************************************
 * accessor_with_properties implementation
 **********************************************************************************************************************/

template <typename _ElementType,
          typename _Restrict,
          typename _Aligment,
          typename _Eviction,
          typename _Prefetch,
          typename _CacheHint>
struct accessor_with_properties
{
  static_assert(!_CUDA_VSTD::is_array<_ElementType>::value,
                "accessor_with_properties: template argument may not be an array type");
  static_assert(!_CUDA_VSTD::is_abstract<_ElementType>::value,
                "accessor_with_properties: template argument may not be an abstract class");

  static constexpr bool _IsConst    = _CUDA_VSTD::is_const_v<_ElementType>;
  static constexpr bool _IsRestrict = _CUDA_VSTD::is_same_v<_Restrict, decltype(aliasing_policy::restrict)>;

  using offset_policy = accessor_with_properties;
  using element_type  = _ElementType;
  using reference     = _CUDA_VSTD::_If<_IsConst, _ElementType, accessor_reference<_ElementType, _Eviction, _Prefetch>>;
  using data_handle_type = _CUDA_VSTD::_If<_IsRestrict, _ElementType* __restrict__, _ElementType*>;

  access_property __prop;

  explicit accessor_with_properties() noexcept = default;

  explicit accessor_with_properties(access_property __prop) noexcept
      : __prop(__prop)
  {}

  // template <typename _OtherElementType,
  //           typename... _OtherProperties,
  //           _CUDA_VSTD::__enable_if_t<_CUDA_VSTD::is_convertible<_OtherElementType (*)[], _ElementType
  //           (*)[]>::value>>
  // _CCCL_HIDE_FROM_ABI _CCCL_DEVICE constexpr accessor_with_properties(
  //   accessor_with_properties<_OtherElementType, _OtherProperties...>) noexcept
  // {}

  _CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE reference access(data_handle_type __p, size_t __i) const noexcept
  {
    // auto __p1 = _CUDA_VSTD::assume_aligned(__p, _Alignment::value);
    //   return cub::ThreadLoad<_Eviction>(__p + __i);
    return __p[__i];
  }

  _CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE data_handle_type offset(data_handle_type __p, size_t __i) const noexcept
  {
    return __p + __i;
  }
};

/***********************************************************************************************************************
 * make_accessor_with_properties
 **********************************************************************************************************************/

template <typename _P, typename... _Ps>
auto __filter_access_properties(_P __prop, _Ps... __properties)
{
  if constexpr (__is_cache_hint<_P>::value)
  {
    return __prop;
  }
  else
  {
    return __filter_access_properties(__properties...);
  }
}

template <typename _ElementType, typename... _UserProperties>
auto make_accessor_with_properties(_UserProperties... __properties) noexcept
{
  using _Restrict  = typename __find_aliasing_policy<_UserProperties...>::__type;
  using _Alignment = typename __find_aligment<alignof(_ElementType), _UserProperties...>::__type;
  using _Eviction  = typename __find_eviction_policy<_UserProperties...>::__type;
  using _Prefetch  = typename __find_prefetch_size<_UserProperties...>::__type;
  using _CacheHint = typename __find_cache_hint_policy<_UserProperties...>::__type;
  static_assert(__type_count_eviction<_UserProperties...>::value <= 1, "Duplicate eviction policy found");
  static_assert(__type_count_aliasing<_UserProperties...>::value <= 1, "Duplicate eviction aliasing policy found");
  static_assert(__type_count_alignment<_UserProperties...>::value <= 1, "Duplicate aligment found");
  static_assert(__type_count_prefetch<_UserProperties...>::value <= 1, "Duplicate prefetch policy found");
  static_assert(__type_count_cache_hint<_UserProperties...>::value <= 1, "Duplicate cache hint policy found");
  return accessor_with_properties<_ElementType, _Restrict, _Alignment, _Eviction, _Prefetch, _CacheHint>(
    __filter_access_properties(__properties..., _CacheHint{}));
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CCCL_STD_VER >= 2014
#endif // _CUDA__MDSPAN_ACCESSOR_WITH_PROPERTIES_H
