//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_CUDA_LOAD_STORE_PROPERTIES
#define __CUDAX_CUDA_LOAD_STORE_PROPERTIES

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>

namespace cuda::experimental
{

/***********************************************************************************************************************
 * Load Behavior
 **********************************************************************************************************************/

enum class MemoryBehavior
{
  ReadOnly,
  ReadWrite,
};

template <MemoryBehavior Value>
using memory_behavior_t = ::cuda::std::integral_constant<MemoryBehavior, Value>;

using read_only_t  = memory_behavior_t<MemoryBehavior::ReadOnly>;
using read_write_t = memory_behavior_t<MemoryBehavior::ReadWrite>;

inline constexpr auto read_only  = read_only_t{};
inline constexpr auto read_write = read_write_t{};

/***********************************************************************************************************************
 * Eviction Policies
 **********************************************************************************************************************/

enum class EvictionPolicyEnum
{
  None,
  First,
  Normal,
  Last,
  LastUse,
  NoAllocation,
};

template <EvictionPolicyEnum Value>
using eviction_policy_t = ::cuda::std::integral_constant<EvictionPolicyEnum, Value>;

using eviction_none_t     = eviction_policy_t<EvictionPolicyEnum::None>;
using eviction_normal_t   = eviction_policy_t<EvictionPolicyEnum::Normal>;
using eviction_first_t    = eviction_policy_t<EvictionPolicyEnum::First>;
using eviction_last_t     = eviction_policy_t<EvictionPolicyEnum::Last>;
using eviction_last_use_t = eviction_policy_t<EvictionPolicyEnum::LastUse>;
using eviction_no_alloc_t = eviction_policy_t<EvictionPolicyEnum::NoAllocation>;

inline constexpr auto eviction_none     = eviction_none_t{};
inline constexpr auto eviction_normal   = eviction_normal_t{};
inline constexpr auto eviction_first    = eviction_first_t{};
inline constexpr auto eviction_last     = eviction_last_t{};
inline constexpr auto eviction_last_use = eviction_last_use_t{};
inline constexpr auto eviction_no_alloc = eviction_no_alloc_t{};

/***********************************************************************************************************************
 * Prefetch Size
 **********************************************************************************************************************/

enum class PrefetchSizeEnum
{
  NoPrefetch,
  Default,
  Bytes64,
  Bytes128,
  Bytes256,
};

template <PrefetchSizeEnum Value>
using prefetch_t = ::cuda::std::integral_constant<PrefetchSizeEnum, Value>;

using no_prefetch_t      = prefetch_t<PrefetchSizeEnum::NoPrefetch>;
using prefetch_default_t = prefetch_t<PrefetchSizeEnum::Default>;
using prefetch_64B_t     = prefetch_t<PrefetchSizeEnum::Bytes64>;
using prefetch_128B_t    = prefetch_t<PrefetchSizeEnum::Bytes128>;
using prefetch_256B_t    = prefetch_t<PrefetchSizeEnum::Bytes256>;

inline constexpr auto no_prefetch      = no_prefetch_t{};
inline constexpr auto prefetch_default = prefetch_default_t{};
inline constexpr auto prefetch_64B     = prefetch_64B_t{};
inline constexpr auto prefetch_128B    = prefetch_128B_t{};
inline constexpr auto prefetch_256B    = prefetch_256B_t{};

/***********************************************************************************************************************
 * Aliasing Policies
 **********************************************************************************************************************/

enum class PtrAliasingPolicyEnum
{
  MayAlias,
  Restrict
};

template <PtrAliasingPolicyEnum Value>
using ptr_aliasing_policy_t = ::cuda::std::integral_constant<PtrAliasingPolicyEnum, Value>;

using ptr_may_alias_t   = ptr_aliasing_policy_t<PtrAliasingPolicyEnum::MayAlias>;
using ptr_no_aliasing_t = ptr_aliasing_policy_t<PtrAliasingPolicyEnum::Restrict>;

inline constexpr auto ptr_may_alias   = ptr_may_alias_t{};
inline constexpr auto ptr_no_aliasing = ptr_no_aliasing_t{};

} // namespace cuda::experimental

#endif // __CUDAX_CUDA_LOAD_STORE_PROPERTIES
