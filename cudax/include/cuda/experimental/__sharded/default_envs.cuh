//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief `default_envs`: derive per-shard environments from structures that
 *        recorded their binding at construction.
 *
 * A `sharded_array` built through a provider already knows, per shard, the
 * stream to order work on and the place its memory lives at. `default_envs`
 * turns that recorded state into the standard per-shard environment family
 * the generic algorithms consume (one `cuda::std::execution::env` per shard,
 * answering `cuda::get_stream` and `cuda::mr::get_memory_resource`) — the
 * same environments `place_group::env` has always manufactured for the
 * per-shard CUB calls.
 *
 * Structures without recorded bindings (pure views, transported descriptors,
 * foreign wrappers) do not get an overload here; they are used through the
 * explicit-environment algorithm overloads, or provide their own
 * `default_envs` found by argument-dependent lookup (which is what makes
 * them model the `self_bound` concept).
 */

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__places/place_group.cuh>
#include <cuda/experimental/__sharded/concepts.cuh>
#include <cuda/experimental/__sharded/cuda_safe_call.cuh>
#include <cuda/experimental/__sharded/sharded_array.cuh>

#include <vector>

namespace cuda::experimental::sharded
{
//! @brief The environment type manufactured for one shard: answers
//! `cuda::get_stream` (the shard's reference stream) and
//! `cuda::mr::get_memory_resource` (a `place_memory_resource` at the shard's
//! data place).
using shard_env_t =
  decltype(places::place_group::env(::cuda::std::declval<const places::data_place&>(), cudaStream_t{}));

//! @brief Per-shard environments of a `sharded_array`, derived from the
//! binding its shards recorded at construction or adoption.
//!
//! The returned environments *borrow* the shards' streams: they are valid
//! for as long as the array's streams are (provider-owned pool streams, or
//! the caller's own streams on adopted arrays).
template <class _Tp>
[[nodiscard]] ::std::vector<shard_env_t> default_envs(const sharded_array<_Tp>& __arr)
{
  ::std::vector<shard_env_t> __envs;
  const ::std::size_t __n = __arr.num_shards();
  __envs.reserve(__n);
  for (const auto __i : each(__n))
  {
    const auto& __s = __arr.shard(__i);
    __envs.push_back(places::place_group::env(__s.place, __s.stream));
  }
  return __envs;
}
} // namespace cuda::experimental::sharded
