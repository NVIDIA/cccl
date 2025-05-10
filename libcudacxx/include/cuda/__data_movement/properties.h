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

#include <cuda/__annotated_ptr/access_property.h>
#include <cuda/__annotated_ptr/associate_access_property.h>
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
 * L1 Eviction Policies
 **********************************************************************************************************************/

enum class _CacheReuseEnum
{
  _Normal,
  _Unchanged,
  _Low,
  _High,
  _NoReuse,
};

template <_CacheReuseEnum _Value>
using __cache_reuse_t = _CUDA_VSTD::integral_constant<_CacheReuseEnum, _Value>;

using __cache_reuse_normal_t    = __cache_reuse_t<_CacheReuseEnum::_Normal>;
using __cache_reuse_unchanged_t = __cache_reuse_t<_CacheReuseEnum::_Unchanged>;
using __cache_reuse_low_t       = __cache_reuse_t<_CacheReuseEnum::_Low>;
using __cache_reuse_high_t      = __cache_reuse_t<_CacheReuseEnum::_High>;
using __cache_no_reuse_t        = __cache_reuse_t<_CacheReuseEnum::_NoReuse>;

inline constexpr auto cache_reuse_normal    = __cache_reuse_normal_t{};
inline constexpr auto cache_reuse_unchanged = __cache_reuse_unchanged_t{};
inline constexpr auto cache_reuse_low       = __cache_reuse_low_t{};
inline constexpr auto cache_reuse_high      = __cache_reuse_high_t{};
inline constexpr auto cache_no_reuse        = __cache_no_reuse_t{};

/***********************************************************************************************************************
 * Prefetch Spatial Locality
 **********************************************************************************************************************/

enum class _L2_PrefetchEnum
{
  _None,
  _Bytes64,
  _Bytes128,
  _Bytes256,
};

template <_L2_PrefetchEnum _Value>
using __l2_prefetch_t = _CUDA_VSTD::integral_constant<_L2_PrefetchEnum, _Value>;

using __L2_prefetch_none_t = __l2_prefetch_t<_L2_PrefetchEnum::_None>;
using __L2_prefetch_64B_t  = __l2_prefetch_t<_L2_PrefetchEnum::_Bytes64>;
using __L2_prefetch_128B_t = __l2_prefetch_t<_L2_PrefetchEnum::_Bytes128>;
using __L2_prefetch_256B_t = __l2_prefetch_t<_L2_PrefetchEnum::_Bytes256>;

inline constexpr auto L2_prefetch_none = __L2_prefetch_none_t{};
inline constexpr auto L2_prefetch_64B  = __L2_prefetch_64B_t{};
inline constexpr auto L2_prefetch_128B = __L2_prefetch_128B_t{};
inline constexpr auto L2_prefetch_256B = __L2_prefetch_256B_t{};

/***********************************************************************************************************************
 * Cache Hint
 **********************************************************************************************************************/

template <typename _AccessProperty>
struct __l2_hint_t : _CUDA_VSTD::bool_constant<!_CUDA_VSTD::is_same_v<_AccessProperty, access_property::global>>
{
  static_assert(::cuda::__is_global_access_property_v<_AccessProperty>, "invalid access property");

  uint64_t __property;

  _CCCL_HIDE_FROM_ABI _CCCL_DEVICE explicit __l2_hint_t(_AccessProperty __property)
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
