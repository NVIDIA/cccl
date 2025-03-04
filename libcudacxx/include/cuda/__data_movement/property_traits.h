//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_CUDA_LOAD_STORE_PROPERTY_TRAITS
#define __CUDAX_CUDA_LOAD_STORE_PROPERTY_TRAITS

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>

#include <cuda/experimental/__mdspan/properties.cuh>

namespace cuda::experimental
{

/***********************************************************************************************************************
 * Load Behavior
 **********************************************************************************************************************/

template <typename>
struct is_memory_behavior : ::cuda::std::false_type
{};

template <MemoryBehavior Value>
struct is_memory_behavior<memory_behavior_t<Value>> : ::cuda::std::true_type
{};

template <typename T>
constexpr bool is_memory_behavior_v = is_memory_behavior<T>::value;

/***********************************************************************************************************************
 * Cache Hint
 **********************************************************************************************************************/

template <typename>
struct is_eviction_policy : ::cuda::std::false_type
{};

template <EvictionPolicyEnum Value>
struct is_eviction_policy<eviction_policy_t<Value>> : ::cuda::std::true_type
{};

template <typename T>
constexpr bool is_eviction_policy_v = is_eviction_policy<T>::value;

/***********************************************************************************************************************
 * Spatial Prefetch
 **********************************************************************************************************************/

template <typename>
struct is_prefetch_spatial : ::cuda::std::false_type
{};

template <PrefetchSpatialEnum Value>
struct is_prefetch_spatial<prefetch_spatial_t<Value>> : ::cuda::std::true_type
{};

template <typename T>
constexpr bool is_prefetch_spatial_v = is_prefetch_spatial<T>::value;

/***********************************************************************************************************************
 * Temporal Prefetch
 **********************************************************************************************************************/

template <typename>
struct is_prefetch_temporal : ::cuda::std::false_type
{};

template <PrefetchSpatialEnum Value>
struct is_prefetch_temporal<prefetch_spatial_t<Value>> : ::cuda::std::true_type
{};

template <typename T>
constexpr bool is_prefetch_temporal_v = is_prefetch_temporal<T>::value;

/***********************************************************************************************************************
 * Aliasing Policies
 **********************************************************************************************************************/

template <typename>
struct is_ptr_aliasing_policy : ::cuda::std::false_type
{};

template <PtrAliasingPolicyEnum Value>
struct is_ptr_aliasing_policy<ptr_aliasing_policy_t<Value>> : ::cuda::std::true_type
{};

template <typename T>
constexpr bool is_ptr_aliasing_policy_v = is_ptr_aliasing_policy<T>::value;

/***********************************************************************************************************************
 * Alignment
 **********************************************************************************************************************/

template <typename>
struct is_alignment : ::cuda::std::false_type
{};

template <size_t AlignBytes>
struct is_alignment<alignment_t<AlignBytes>> : ::cuda::std::true_type
{};

template <typename T>
constexpr bool is_alignment_v = is_alignment<T>::value;

} // namespace cuda::experimental

#endif // __CUDAX_CUDA_LOAD_STORE_PROPERTY_TRAITS
