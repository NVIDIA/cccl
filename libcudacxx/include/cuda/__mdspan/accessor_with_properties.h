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

#  include <cub/thread/thread_load.cuh>
#  include <cub/thread/thread_store.cuh>

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

/***********************************************************************************************************************
 * Eviction Policies
 **********************************************************************************************************************/

enum class _EvictionPolicyEnum
{
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

/// @brief Cache eviction policies determine the order in which cache entries are removed when the cache reaches its
///        capacity
namespace eviction_policy
{

/// @brief Evict first. Data will likely be evicted when cache eviction is required. This policy is suitable for
///        streaming data
constexpr auto first = __eviction_policy_constant<_EvictionPolicyEnum::_First>;

/// @brief Default eviction policy. It maps to a standard memory access
constexpr auto normal = __eviction_policy_constant<_EvictionPolicyEnum::_Normal>;

/// @brief Evict last. Data will likely be evicted only after other data with 'evict_normal' or 'evict_first' eviction
///        priotity is already evicted. This policy is suitable for persistentdata
constexpr auto last = __eviction_policy_constant<_EvictionPolicyEnum::_Last>;

/// @brief Last use. Data that is read can be invalidated even if dirty
constexpr auto last_use = __eviction_policy_constant<_EvictionPolicyEnum::_LastUse>;

/// @brief No allocation. Do not allocate data to cache. This policy is suitable for streaming data
constexpr auto no_allocation = __eviction_policy_constant<_EvictionPolicyEnum::_NoAllocation>;

}; // namespace eviction_policy

/***********************************************************************************************************************
 * Memory Consistency Scope
 **********************************************************************************************************************/

enum class _MemoryConsistencyScope
{
  _None,
  _Cta,
  _Cluster,
  _Gpu,
  _System
};

template <_MemoryConsistencyScope _Value>
using __memory_consistency_scope_constant_t = _CUDA_VSTD::integral_constant<_MemoryConsistencyScope, _Value>;

template <_MemoryConsistencyScope _Value>
constexpr auto __memory_consistency_scope = __memory_consistency_scope_constant_t<_Value>{};

template <typename>
struct __is_memory_consistency_scope : _CUDA_VSTD::false_type
{};

template <_MemoryConsistencyScope _Value>
struct __is_memory_consistency_scope<__memory_consistency_scope_constant_t<_Value>> : _CUDA_VSTD::true_type
{};

namespace memory_consistency_scope
{

constexpr auto none    = __memory_consistency_scope<_MemoryConsistencyScope::_None>;
constexpr auto cta     = __memory_consistency_scope<_MemoryConsistencyScope::_Cta>;
constexpr auto cluster = __memory_consistency_scope<_MemoryConsistencyScope::_Cluster>;
constexpr auto gpu     = __memory_consistency_scope<_MemoryConsistencyScope::_Gpu>;
constexpr auto system  = __memory_consistency_scope<_MemoryConsistencyScope::_System>;

}; // namespace memory_consistency_scope

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

template <_CUDA_VSTD::size_t __AlignBytes>
struct __is_alignment<aligned_size_t<__AlignBytes>> : _CUDA_VSTD::true_type
{};

/***********************************************************************************************************************
 * Cache Hints
 **********************************************************************************************************************/

template <typename _T>
struct __is_cache_hint
    : _CUDA_VSTD::bool_constant<_CUDA_VSTD::is_same<_T, access_property::streaming>::value
                                || _CUDA_VSTD::is_same<_T, access_property::persisting>::value
                                || _CUDA_VSTD::is_same<_T, access_property::normal>::value
                                || _CUDA_VSTD::is_same<_T, access_property::global>::value
                                || _CUDA_VSTD::is_same<_T, access_property::shared>::value>
{};

/***********************************************************************************************************************
 * Find Duplicate Utilities
 **********************************************************************************************************************/

template <template <typename> class _Predicate>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_count_if
{
  template <class... _Ts>
  using __call _LIBCUDACXX_NODEBUG_TYPE = _CUDA_VSTD::integral_constant<int, (_Predicate<_Ts>::value + ... + 0)>;
};

template <class... _Ts>
using __type_count_eviction = _CUDA_VSTD::__type_call<__type_count_if<__is_eviction_policy>, _Ts...>;

template <class... _Ts>
using __type_count_memory_consistency = _CUDA_VSTD::__type_call<__type_count_if<__is_memory_consistency_scope>, _Ts...>;

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

template <template <typename> class _Predicate, typename _DefaultValue, typename... _UserProperties>
struct __find_property
{
  using __ret = _CUDA_VSTD::__type_find_if<_CUDA_VSTD::__type_list<_UserProperties...>, __predicate_call<_Predicate>>;

  using __type = _CUDA_VSTD::__type_front<_CUDA_VSTD::__type_concat<__ret, _CUDA_VSTD::__type_list<_DefaultValue>>>;
};

template <typename... _UserProperties>
using __find_eviction_policy =
  __find_property<__is_eviction_policy, decltype(eviction_policy::normal), _UserProperties...>;

template <typename... _UserProperties>
using __find_memory_consistency_scope =
  __find_property<__is_memory_consistency_scope, decltype(memory_consistency_scope::none), _UserProperties...>;

template <typename... _UserProperties>
using __find_prefetch_size =
  __find_property<__is_prefetch_policy, decltype(prefetch_size::no_prefetch), _UserProperties...>;

template <_CUDA_VSTD::size_t _DefaultAlignment, typename... _UserProperties>
using __find_alignment = __find_property<__is_alignment, aligned_size_t<_DefaultAlignment>, _UserProperties...>;

template <typename... _UserProperties>
using __find_aliasing_policy =
  __find_property<__is_aliasing_policy, decltype(aliasing_policy::restrict), _UserProperties...>;

template <typename... _UserProperties>
using __find_cache_hint_policy = __find_property<__is_cache_hint, access_property::global, _UserProperties...>;

/***********************************************************************************************************************
 * accessor_with_properties implementation
 **********************************************************************************************************************/

template <typename...>
struct __always_false : _CUDA_VSTD::false_type
{};

// placeholder
template <typename T>
T* assume_aligned(T* __ptr, size_t)
{
  return __ptr;
};

template <bool _IsConst, _EvictionPolicyEnum _Eviction, _MemoryConsistencyScope _Scope>
constexpr auto __to_cub_load_enum()
{
  // Current CUB limitations
  constexpr auto __current_eviction_policy           = __eviction_policy_constant<_Eviction>;
  constexpr auto __current_memory_consistency_scopey = __memory_consistency_scope<_Scope>;
  static_assert(__current_eviction_policy == eviction_policy::last,
                "eviction_policy::last is currently unsupported eviction policy");
  static_assert(__current_eviction_policy == eviction_policy::last_use,
                "eviction_policy::last_use is currently unsupported eviction policy");
  static_assert(__current_eviction_policy == eviction_policy::no_allocation,
                "eviction_policy::no_allocation is currently unsupported eviction policy");
  static_assert(__current_memory_consistency_scopey == memory_consistency_scope::cluster,
                "memory_consistency_scope::cluster is currently unsupported memory consistency scope");
  if constexpr (_IsConst && _Eviction == _EvictionPolicyEnum::_Normal && _Scope == _MemoryConsistencyScope::_None)
  {
    return cub::LOAD_LDG;
  }
  else if constexpr (_Eviction == _EvictionPolicyEnum::_Normal && _Scope == _MemoryConsistencyScope::_None)
  {
    return cub::LOAD_DEFAULT;
  }
  else if constexpr (_Eviction == _EvictionPolicyEnum::_Normal && _Scope == _MemoryConsistencyScope::_Cta)
  {
    return cub::LOAD_CA;
  }
  else if constexpr (_Eviction == _EvictionPolicyEnum::_Normal && _Scope == _MemoryConsistencyScope::_Gpu)
  {
    return cub::LOAD_CG;
  }
  else if constexpr (_Eviction == _EvictionPolicyEnum::_Normal && _Scope == _MemoryConsistencyScope::_System)
  {
    return cub::LOAD_CV;
  }
  else if constexpr (_Eviction == _EvictionPolicyEnum::_First && _Scope == _MemoryConsistencyScope::_None)
  {
    return cub::LOAD_CS;
  }
  else
  {
    return cub::LOAD_DEFAULT;
    // static_assert(__always_false<_EvictionPolicyEnum, _MemoryConsistencyScope>::value,
    //               "Unsupported Eviction Policy/Memory Consistency Scope combination for load operation");
  }
}

template <_EvictionPolicyEnum _Eviction, _MemoryConsistencyScope _Scope>
constexpr auto __to_cub_store_enum()
{
  // Current CUB limitations
  constexpr auto __current_eviction_policy           = __eviction_policy_constant<_Eviction>;
  constexpr auto __current_memory_consistency_scopey = __memory_consistency_scope<_Scope>;
  static_assert(__current_eviction_policy == eviction_policy::last,
                "eviction_policy::last is currently unsupported eviction policy");
  static_assert(__current_eviction_policy == eviction_policy::last_use,
                "eviction_policy::last_use is currently unsupported eviction policy");
  static_assert(__current_eviction_policy == eviction_policy::no_allocation,
                "eviction_policy::no_allocation is currently unsupported eviction policy");
  static_assert(__current_memory_consistency_scopey == memory_consistency_scope::cluster,
                "memory_consistency_scope::cluster is currently unsupported memory consistency scope");
  if constexpr (_Eviction == _EvictionPolicyEnum::_Normal && _Scope == _MemoryConsistencyScope::_None)
  {
    return cub::STORE_DEFAULT;
  }
  else if constexpr (_Eviction == _EvictionPolicyEnum::_Normal && _Scope == _MemoryConsistencyScope::_Cta)
  {
    return cub::STORE_WB;
  }
  else if constexpr (_Eviction == _EvictionPolicyEnum::_Normal && _Scope == _MemoryConsistencyScope::_Gpu)
  {
    return cub::STORE_CG;
  }
  else if constexpr (_Eviction == _EvictionPolicyEnum::_Normal && _Scope == _MemoryConsistencyScope::_System)
  {
    return cub::STORE_WT;
  }
  else if constexpr (_Eviction == _EvictionPolicyEnum::_First && _Scope == _MemoryConsistencyScope::_None)
  {
    return cub::STORE_CS;
  }
  else
  {
    return cub::STORE_DEFAULT;
    // static_assert(__always_false<_EvictionPolicyEnum, _MemoryConsistencyScope>::value,
    //               "Unsupported Eviction Policy/Memory Consistency Scope combination for store operation");
  }
}

/***********************************************************************************************************************
 * accessor_with_properties foward declaration
 **********************************************************************************************************************/

template <typename _ElementType,
          typename _Restrict,
          typename _Alignment,
          typename _Eviction,
          typename _Scope,
          typename _Prefetch,
          typename _CacheHint>
struct accessor_with_properties;

/***********************************************************************************************************************
 * accessor_reference
 **********************************************************************************************************************/

template <typename _ElementType,
          typename _Restrict,
          typename _Alignment,
          typename _Eviction,
          typename _Scope,
          typename _Prefetch,
          typename _CacheHint>
class accessor_reference
{
  static constexpr bool _IsRestrict = _CUDA_VSTD::is_same_v<_Restrict, decltype(aliasing_policy::restrict)>;

  using __pointer_type = _CUDA_VSTD::_If<_IsRestrict, _ElementType* __restrict__, _ElementType*>;

  __pointer_type __p;

  friend class accessor_with_properties<_ElementType, _Restrict, _Alignment, _Eviction, _Scope, _Prefetch, _CacheHint>;

public:
  explicit accessor_reference() noexcept = default;

  accessor_reference(accessor_reference&&) = delete;

  accessor_reference& operator=(accessor_reference&&) = delete;

  _CCCL_HIDE_FROM_ABI accessor_reference(const accessor_reference&) noexcept = default;

  _CCCL_HIDE_FROM_ABI _CCCL_DEVICE accessor_reference& operator=(const accessor_reference& __x) noexcept
  {
    return operator=(static_cast<_ElementType>(__x));
  }

  _CCCL_HIDE_FROM_ABI _CCCL_DEVICE accessor_reference& operator=(_ElementType __x) noexcept
  {
    constexpr auto __cub_enum = __to_cub_store_enum<_Eviction, _Scope>();
    return cub::ThreadStore<__cub_enum>(__p, __x);
  }

  _CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE operator _ElementType() const noexcept
  {
    constexpr bool __is_const_elem = _CUDA_VSTD::is_const_v<_ElementType>;
    constexpr auto __cub_enum      = __to_cub_load_enum<__is_const_elem, _Eviction, _Scope>();
    return cub::ThreadLoad<__cub_enum>(__p);
  }

private:
  _CCCL_HIDE_FROM_ABI _CCCL_DEVICE explicit accessor_reference(__pointer_type __p_) noexcept
      : __p{__p_}
  {}
};

/***********************************************************************************************************************
 * accessor_with_properties implementation
 **********************************************************************************************************************/

template <typename _ElementType,
          typename _Restrict,
          typename _Alignment,
          typename _Eviction,
          typename _Scope,
          typename _Prefetch,
          typename _CacheHint>
class accessor_with_properties
{
  static constexpr bool _IsConst    = _CUDA_VSTD::is_const_v<_ElementType>;
  static constexpr bool _IsRestrict = _CUDA_VSTD::is_same_v<_Restrict, decltype(aliasing_policy::restrict)>;

public:
  static_assert(!_CUDA_VSTD::is_array_v<_ElementType>,
                "accessor_with_properties: template argument may not be an array type");
  static_assert(!_CUDA_VSTD::is_abstract_v<_ElementType>,
                "accessor_with_properties: template argument may not be an abstract class");

  using offset_policy = accessor_with_properties;
  using element_type  = _ElementType;
  using reference =
    _CUDA_VSTD::_If<_IsConst,
                    _ElementType,
                    accessor_reference<_ElementType, _Restrict, _Alignment, _Eviction, _Scope, _Prefetch, _CacheHint>>;
  using data_handle_type = _CUDA_VSTD::_If<_IsRestrict, _ElementType* __restrict__, _ElementType*>;

  explicit accessor_with_properties() noexcept = default;

  explicit accessor_with_properties(access_property __prop) noexcept
      : __prop{__prop}
  {}

  template <typename _OtherElementType,
            typename... _OtherProperties,
            _CUDA_VSTD::__enable_if_t<_CUDA_VSTD::is_convertible<_OtherElementType (*)[], _ElementType (*)[]>::value>>
  _CCCL_HIDE_FROM_ABI _CCCL_DEVICE constexpr accessor_with_properties(
    accessor_with_properties<_OtherElementType, _OtherProperties...>) noexcept
  {}

  _CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE reference access(data_handle_type __p, size_t __i) const noexcept
  {
    auto __p1 = /*_CUDA_VSTD::*/ assume_aligned(__p, _Alignment::value);
    if constexpr (_IsConst)
    {
      constexpr auto __cub_enum = __to_cub_load_enum<_IsConst, _Eviction, _Scope>();
      return cub::ThreadLoad<__cub_enum>(__p1 + __i);
    }
    else
    {
      return reference{__p1 + __i};
    }
  }

  _CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE data_handle_type offset(data_handle_type __p, size_t __i) const noexcept
  {
    auto __p1 = /*_CUDA_VSTD::*/ assume_aligned(__p, _Alignment::value);
    return __p1 + __i;
  }

private:
  access_property __prop;
};

/***********************************************************************************************************************
 * make_accessor_with_properties
 **********************************************************************************************************************/

template <typename _P, typename... _Ps>
auto __filter_access_property(_P __prop, _Ps... __properties)
{
  if constexpr (__is_cache_hint<_P>::value)
  {
    return __prop;
  }
  else
  {
    return __filter_access_property(__properties...);
  }
}

template <typename _ElementType, typename... _UserProperties>
auto make_accessor_with_properties(_UserProperties... __properties) noexcept
{
  using _Restrict  = typename __find_aliasing_policy<_UserProperties...>::__type;
  using _Alignment = typename __find_alignment<alignof(_ElementType), _UserProperties...>::__type;
  using _Eviction  = typename __find_eviction_policy<_UserProperties...>::__type;
  using _Scope     = typename __find_memory_consistency_scope<_UserProperties...>::__type;
  using _Prefetch  = typename __find_prefetch_size<_UserProperties...>::__type;
  using _CacheHint = typename __find_cache_hint_policy<_UserProperties...>::__type;
  static_assert(__type_count_aliasing<_UserProperties...>::value <= 1, "Duplicate eviction aliasing policy found");
  static_assert(__type_count_alignment<_UserProperties...>::value <= 1, "Duplicate alignment found");
  static_assert(__type_count_eviction<_UserProperties...>::value <= 1, "Duplicate eviction policy found");
  static_assert(__type_count_memory_consistency<_UserProperties...>::value <= 1,
                "Duplicate memory consistency scope found");
  static_assert(__type_count_prefetch<_UserProperties...>::value <= 1, "Duplicate prefetch policy found");
  static_assert(__type_count_cache_hint<_UserProperties...>::value <= 1, "Duplicate cache hint policy found");
  return accessor_with_properties<_ElementType, _Restrict, _Alignment, _Eviction, _Scope, _Prefetch, _CacheHint>(
    __filter_access_property(__properties..., _CacheHint{}));
}

template <typename T>
using streaming_accessor = decltype(make_accessor_with_properties<T>(eviction_policy::first));

template <typename T>
using cache_all_accessor = decltype(make_accessor_with_properties<T>(memory_consistency_scope::gpu));

template <typename T>
using cache_global_accessor = decltype(make_accessor_with_properties<T>(memory_consistency_scope::cta));

template <typename T>
using cache_invalidation_accessor = decltype(make_accessor_with_properties<T>(memory_consistency_scope::system));

template <typename T>
using read_only_accessor = decltype(make_accessor_with_properties<const T>());

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CCCL_STD_VER >= 2014
#endif // _CUDA__MDSPAN_ACCESSOR_WITH_PROPERTIES_H
