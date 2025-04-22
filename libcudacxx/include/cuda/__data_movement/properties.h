//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___DATA_MOVEMENT_PROPERTIES_H
#define _CUDA___DATA_MOVEMENT_PROPERTIES_H

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/annotated_ptr>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/cstdint>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_DEVICE

/***********************************************************************************************************************
 * Load Behavior
 **********************************************************************************************************************/

enum class _MemoryAccess
{
  _ReadOnly,
  _ReadWrite,
};

template <_MemoryAccess _Value>
using __memory_access_t = _CUDA_VSTD::integral_constant<_MemoryAccess, _Value>;

using __read_only_t  = __memory_access_t<_MemoryAccess::_ReadOnly>;
using __read_write_t = __memory_access_t<_MemoryAccess::_ReadWrite>;

inline constexpr auto read_only  = __read_only_t{};
inline constexpr auto read_write = __read_write_t{};

/***********************************************************************************************************************
 * Eviction Policies
 **********************************************************************************************************************/

enum class _EvictionPolicyEnum
{
  _None,
  _Normal,
  _Unchanged,
  _First,
  _Last,
  _NoAllocation,
};

template <_EvictionPolicyEnum _Value>
using __eviction_policy_t = _CUDA_VSTD::integral_constant<_EvictionPolicyEnum, _Value>;

using __eviction_none_t      = __eviction_policy_t<_EvictionPolicyEnum::_None>;
using __eviction_normal_t    = __eviction_policy_t<_EvictionPolicyEnum::_Normal>;
using __eviction_unchanged_t = __eviction_policy_t<_EvictionPolicyEnum::_Unchanged>;
using __eviction_first_t     = __eviction_policy_t<_EvictionPolicyEnum::_First>;
using __eviction_last_t      = __eviction_policy_t<_EvictionPolicyEnum::_Last>;
using __eviction_no_alloc_t  = __eviction_policy_t<_EvictionPolicyEnum::_NoAllocation>;

inline constexpr auto eviction_none      = __eviction_none_t{};
inline constexpr auto eviction_normal    = __eviction_normal_t{};
inline constexpr auto eviction_unchanged = __eviction_unchanged_t{};
inline constexpr auto eviction_first     = __eviction_first_t{};
inline constexpr auto eviction_last      = __eviction_last_t{};
inline constexpr auto eviction_no_alloc  = __eviction_no_alloc_t{};

/***********************************************************************************************************************
 * Prefetch Spatial Locality
 **********************************************************************************************************************/

enum class _PrefetchSpatialEnum
{
  _None,
  _Bytes64,
  _Bytes128,
  _Bytes256,
};

template <_PrefetchSpatialEnum _Value>
using __prefetch_spatial_t = _CUDA_VSTD::integral_constant<_PrefetchSpatialEnum, _Value>;

using __prefetch_spatial_none_t = __prefetch_spatial_t<_PrefetchSpatialEnum::_None>;
using __prefetch_64B_t          = __prefetch_spatial_t<_PrefetchSpatialEnum::_Bytes64>;
using __prefetch_128B_t         = __prefetch_spatial_t<_PrefetchSpatialEnum::_Bytes128>;
using __prefetch_256B_t         = __prefetch_spatial_t<_PrefetchSpatialEnum::_Bytes256>;

inline constexpr auto prefetch_spatial_none = __prefetch_spatial_none_t{};
inline constexpr auto prefetch_64B          = __prefetch_64B_t{};
inline constexpr auto prefetch_128B         = __prefetch_128B_t{};
inline constexpr auto prefetch_256B         = __prefetch_256B_t{};

/***********************************************************************************************************************
 * Cache Hint
 **********************************************************************************************************************/

// template <bool _Enabled = true>
// struct _CacheHint
//{
//   uint64_t __property;
// };
//
// inline constexpr auto __no_cache_hint = _CacheHint<false>{};

template <typename _AccessProperty>
struct _CacheHint : _CUDA_VSTD::bool_constant<!_CUDA_VSTD::is_same_v<_AccessProperty, access_property::global>>
{
  // TODO: remove comment after PR #4503
  // static_assert(::cuda::__ap_detail::is_global_access_property<_AccessProperty>, "invalid access property");

  uint64_t __property;

  _CCCL_HIDE_FROM_ABI _CCCL_DEVICE explicit _CacheHint(_AccessProperty __property)
      : __property(static_cast<uint64_t>(access_property{__property}))
  {}
};

#if __cccl_ptx_isa >= 830
inline constexpr size_t __max_ptx_access_size = 16;
#else
inline constexpr size_t __max_ptx_access_size = 8;
#endif

_LIBCUDACXX_END_NAMESPACE_CUDA_DEVICE

#endif // _CUDA___DATA_MOVEMENT_PROPERTIES_H
