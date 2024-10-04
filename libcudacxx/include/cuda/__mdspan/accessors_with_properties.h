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
//           EvictionPolicy _Eviction = EvictionPolicy::Default,
//           PrefetchSize _Prefetch   = PrefetchSize::NoPrefetch,
//           typename _CacheHint      = ::cuda::access_property::normal,
//           typename _Enable         = void>
// struct accessor_with_properties;
//
///***********************************************************************************************************************
// * accessor_reference
// **********************************************************************************************************************/
//
// template <class _ElementType, EvictionPolicy _Eviction, PrefetchSize _Prefetch, typename _CacheHint>
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
// template <class _ElementType, EvictionPolicy _Eviction, PrefetchSize _Prefetch, typename _CacheHint>
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
//  using reference     = ::cuda::accessor_reference<_ElementType, _Eviction, _Prefetch, _CacheHint>;
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
constexpr auto __eviction_policy_constant = _CUDA_VSTD::integral_constant<_EvictionPolicyEnum, _Value>{};

template <typename>
struct __is_eviction_policy : _CUDA_VSTD::false_type
{};

template <_EvictionPolicyEnum _Value>
struct __is_eviction_policy<_CUDA_VSTD::integral_constant<_EvictionPolicyEnum, _Value>> : _CUDA_VSTD::true_type
{};

namespace EvictionPolicy
{

constexpr auto Default      = __eviction_policy_constant<_EvictionPolicyEnum::_Default>;
constexpr auto First        = __eviction_policy_constant<_EvictionPolicyEnum::_First>;
constexpr auto Normal       = __eviction_policy_constant<_EvictionPolicyEnum::_Normal>;
constexpr auto Last         = __eviction_policy_constant<_EvictionPolicyEnum::_Last>;
constexpr auto LastUse      = __eviction_policy_constant<_EvictionPolicyEnum::_LastUse>;
constexpr auto NoAllocation = __eviction_policy_constant<_EvictionPolicyEnum::_NoAllocation>;

}; // namespace EvictionPolicy

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

namespace PrefetchSize
{

constexpr auto NoPrefetch = __prefetch_constant<_PrefetchSizeEnum::_NoPrefetch>;
constexpr auto Bytes64    = __prefetch_constant<_PrefetchSizeEnum::_Bytes64>;
constexpr auto Bytes128   = __prefetch_constant<_PrefetchSizeEnum::_Bytes128>;
constexpr auto Bytes256   = __prefetch_constant<_PrefetchSizeEnum::_Bytes256>;

}; // namespace PrefetchSize

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

namespace AliasingPolicy
{

constexpr auto Restrict = __aliasing_constant<_AliasingPolicyEnum::_Restrict>;
constexpr auto MayAlias = __aliasing_constant<_AliasingPolicyEnum::_MayAlias>;

}; // namespace AliasingPolicy

/***********************************************************************************************************************
 * Alignment
 **********************************************************************************************************************/

template <typename>
struct __is_alignment : _CUDA_VSTD::false_type
{};

template <size_t __AlignBytes>
struct __is_alignment<::cuda::aligned_size_t<__AlignBytes>> : _CUDA_VSTD::true_type
{};

/***********************************************************************************************************************
 * Find Duplicates
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
using __type_count_cache_hints = _CUDA_VSTD::__type_call<__type_count_if<__is_cache_hints>, _Ts...>;

/***********************************************************************************************************************
 * Find Duplicates
 **********************************************************************************************************************/

struct __is_alignment_call
{
  template <typename _T>
  using __call = _CUDA_VSTD::bool_constant<__is_alignment_impl<_T>::value>;
};

template <typename... _Ts>
struct __find_type
{
  template <typename _Property>
  using __call = _CUDA_VSTD::bool_constant<(_CUDA_VSTD::is_same_v<_Property, _Ts> || ...)>;
};

template <typename _TypeList, typename... _UserProperties>
struct __find_property;

template <typename... _Ts, typename... _UserProperties>
struct __find_property<_CUDA_VSTD::__type_list<_Ts...>, _UserProperties...>
{
  using type1 = _CUDA_VSTD::__type_find_if<_CUDA_VSTD::__type_list<_UserProperties...>, __find_type<_Ts...>>;

  using type = _CUDA_VSTD::__type_front<_CUDA_VSTD::__type_concat<type1, _CUDA_VSTD::__type_list<void>>>;
};

template <typename... _UserProperties>
struct __find_eviction_policy
{
  using __find_result =
    __find_property<_CUDA_VSTD::__type_list<EvictionPolicy::Default,
                                            EvictionPolicy::First,
                                            EvictionPolicy::Normal,
                                            EvictionPolicy::Last,
                                            EvictionPolicy::LastUse,
                                            EvictionPolicy::NoAllocation>,
                    _UserProperties...>;

  using type = _CUDA_VSTD::_If<_CUDA_VSTD::is_void_v<__find_result>, EvictionPolicy::Default, __find_result>;
};

template <typename... _UserProperties>
struct __find_prefetch_size
{
  using __find_result = __find_property<
    _CUDA_VSTD::
      __type_list<PrefetchSize::NoPrefetch, PrefetchSize::Bytes64, PrefetchSize::Bytes128, PrefetchSize::Bytes256>,
    _UserProperties...>;

  using type = _CUDA_VSTD::_If<_CUDA_VSTD::is_void_v<__find_result>, PrefetchSize::NoPrefetch, __find_result>;
};

template <typename... _UserProperties>
struct __find_cache_hint_policy
{
  using __find_result =
    typename __find_property<_CUDA_VSTD::__type_list<access_property::shared,
                                                     access_property::global,
                                                     access_property::persisting,
                                                     access_property::streaming,
                                                     access_property::normal>,
                             _UserProperties...>::type;
  using type = _CUDA_VSTD::_If<_CUDA_VSTD::is_void_v<__find_result>, access_property::global, __find_result>;
};

template <typename... _UserProperties>
struct __find_aliasing_policy
{
  using __find_result =
    __find_property<_CUDA_VSTD::__type_list<AliasingPolicy::Restrict, AliasingPolicy::MayAlias>, _UserProperties...>;

  using type = _CUDA_VSTD::_If<_CUDA_VSTD::is_void_v<__find_result>, AliasingPolicy::Restrict, __find_result>;
};

template <typename DefaultAlignment, typename... _UserProperties>
struct __find_aligment
{
  using __find_result = _CUDA_VSTD::__type_front<
    _CUDA_VSTD::__type_find_if<_CUDA_VSTD::__type_list<_UserProperties...>, __is_alignment_call>>;

  using type = _CUDA_VSTD::_If<_CUDA_VSTD::is_void_v<__find_result>, DefaultAlignment, __find_result>;
};

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
  static constexpr bool _IsRestrict = _CUDA_VSTD::is_same_v<_Restrict, AliasingPolicy::Restrict>;

  using offset_policy = accessor_with_properties;
  using element_type  = _ElementType;
  using reference     = _CUDA_VSTD::_If<_IsConst, _ElementType, accessor_reference<_ElementType, _Eviction, _Prefetch>>;
  using data_handle_type = _CUDA_VSTD::_If<_IsRestrict, _ElementType * _CCCL_RESTRICT, _ElementType*>;

  access_property _prop;

  explicit accessor_with_properties() noexcept = default;

  explicit accessor_with_properties(access_property _prop) noexcept
      : _prop(_prop)
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

template <typename _ElementType, typename... _UserProperties>
auto make_accessor_with_properties(_UserProperties... properties) noexcept
{
  using _Restrict  = __find_aliasing_policy<_UserProperties...>;
  using _Alignment = __find_aligment<cuda::aligned_size_t<alignof(_ElementType)>, _UserProperties...>;
  using _Eviction  = __find_eviction_policy<_UserProperties...>;
  using _Prefetch  = __find_prefetch_size<_UserProperties...>;
  using _CacheHint = typename __find_cache_hint_policy<_UserProperties...>::type;
  static_assert(__type_count_eviction<_UserProperties...>::value <= 1, "");

  if constexpr (std::is_same_v<_CacheHint, cuda::access_property::global>)
  {
    return accessor_with_properties<_ElementType, _Restrict, _Alignment, _Eviction, _Prefetch, _CacheHint>();
  }
  else
  {
    return 3; // accessor_with_properties<_ElementType, _Restrict, _Alignment, _Eviction, _Prefetch,
  }
  //    _CacheHint>(get_cache_hint(properties...));
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CCCL_STD_VER >= 2014
#endif // _CUDA__MDSPAN_ACCESSOR_WITH_PROPERTIES_H
