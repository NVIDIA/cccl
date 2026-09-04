//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Counting over sharded arrays: each place runs the device-scope
 *        primitive (CUB `DeviceReduce::TransformReduce` with a 0/1 transform)
 *        on its shard, then the per-place counts are summed.
 *
 * Counting is read-only: it never mutates shard sizes, so it is available on
 * every sharded array, including contiguous (`allocate_contiguous`) ones.
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

#include <cub/device/device_reduce.cuh>

#include <cuda/std/functional>

#include <cuda/experimental/__places/place_group.cuh>
#include <cuda/experimental/__sharded/composition.cuh>
#include <cuda/experimental/__sharded/concepts.cuh>
#include <cuda/experimental/__sharded/cuda_safe_call.cuh>
#include <cuda/experimental/__sharded/default_envs.cuh>
#include <cuda/experimental/__sharded/pinned_staging.cuh>
#include <cuda/experimental/__sharded/sharded_array.cuh>
#include <cuda/experimental/__sharded/stream_scope.cuh>

#include <algorithm>
#include <vector>

#include <cuda_runtime.h>

namespace cuda::experimental::sharded
{
namespace reserved
{
/// @brief Maps an element to 1 when the predicate holds, 0 otherwise.
template <typename _Tp, typename _Pred>
struct count_transform_fn
{
  _Pred pred;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE size_t operator()(_Tp val) const
  {
    return pred(val) ? size_t{1} : size_t{0};
  }
};

/// @brief Equality with a fixed value.
template <typename _Tp>
struct equals_value_fn
{
  _Tp value;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE bool operator()(_Tp val) const
  {
    return val == value;
  }
};
} // namespace reserved

// ============================================================================
// Concept-generic tier: count over any sharded_view
// ============================================================================

/**
 * @brief Count elements satisfying @p pred over any `sharded_view`:
 * per-shard `cub::DeviceReduce::TransformReduce` on each shard's environment,
 * host sum of the per-shard counts. SYNCHRONOUS-only (host combine): refuses
 * at entry under `sync_policy::forbid` and under capture. Staging via the
 * call environment's resource when present, the pinned arena otherwise.
 */
_CCCL_TEMPLATE(class _S, class _Envs, class _Pred, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  sharded_view<::cuda::std::remove_cvref_t<_S>> _CCCL_AND sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Envs>>)
[[nodiscard]] _CCCL_HOST_API size_t
count_if(const _S& data, const _Envs& envs, _Pred pred, const _CallEnv& call_env = {})
{
  using elem_t                   = view_element_t<_S>;
  const ::std::size_t num_shards = reserved::__shard_count(data);
  if (reserved::__env_count(envs) < num_shards)
  {
    _CCCL_THROW(::std::invalid_argument, "sharded::count_if: fewer environments than shards");
  }
  if (num_shards == 0)
  {
    return 0;
  }

  // Refusals first, before any CUDA call: this form synchronizes.
  require_sync_allowed(call_env, "sharded::count_if (synchronous form)");
  places::check_not_capturing(nullptr, "sharded::count_if");
  for (const auto g : each(num_shards))
  {
    places::check_not_capturing(::cuda::get_stream(envs[g]).get(), "sharded::count_if");
  }

  constexpr bool __env_has_mr = ::cuda::std::execution::__queryable_with<_CallEnv, ::cuda::mr::get_memory_resource_t>
                             || ::cuda::mr::__has_member_get_resource<_CallEnv>;
  size_t* h_counts            = nullptr;
  if constexpr (__env_has_mr)
  {
    auto staging_mr = ::cuda::mr::get_memory_resource(call_env);
    h_counts        = static_cast<size_t*>(staging_mr.allocate_sync(num_shards * sizeof(size_t), alignof(size_t)));
  }
  else
  {
    h_counts = static_cast<size_t*>(reserved::__pinned_staging(num_shards * sizeof(size_t)));
  }
  ::std::fill(h_counts, h_counts + num_shards, size_t{0});

  for (const auto g : each(num_shards))
  {
    const auto& s = data.shard(g);
    if (s.size == 0)
    {
      continue;
    }
    const auto& env                       = envs[g];
    const ::cuda::stream_ref shard_stream = ::cuda::get_stream(env);
    stream_scope scope(shard_stream.get());
    auto mr       = ::cuda::mr::get_memory_resource(env);
    size_t* d_out = static_cast<size_t*>(mr.allocate(shard_stream, sizeof(size_t), alignof(size_t)));
    cuda_safe_call(cub::DeviceReduce::TransformReduce(
      s.data,
      d_out,
      s.size,
      ::cuda::std::plus<size_t>{},
      reserved::count_transform_fn<elem_t, _Pred>{pred},
      size_t{0},
      env));
    cuda_safe_call(cudaMemcpyAsync(&h_counts[g], d_out, sizeof(size_t), cudaMemcpyDeviceToHost, shard_stream.get()));
    mr.deallocate(shard_stream, d_out, sizeof(size_t), alignof(size_t)); // stream-ordered, after the copy
  }

  barrier(envs);

  size_t total = 0;
  for (const auto g : each(num_shards))
  {
    total += h_counts[g];
  }

  if constexpr (__env_has_mr)
  {
    auto staging_mr = ::cuda::mr::get_memory_resource(call_env);
    staging_mr.deallocate_sync(h_counts, num_shards * sizeof(size_t), alignof(size_t));
  }
  // (arena staging is cached; nothing to release)

  return total;
}

/// @brief Count elements satisfying @p pred (generic, self-bound).
_CCCL_TEMPLATE(class _S, class _Pred, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  self_bound<::cuda::std::remove_cvref_t<_S>> _CCCL_AND(!sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Pred>>))
[[nodiscard]] _CCCL_HOST_API size_t count_if(const _S& data, _Pred pred, const _CallEnv& call_env = {})
{
  const auto envs = default_envs(data);
  return sharded::count_if(data, envs, pred, call_env);
}

/// @brief Count elements equal to @p value (generic).
_CCCL_TEMPLATE(class _S, class _Envs, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  sharded_view<::cuda::std::remove_cvref_t<_S>> _CCCL_AND sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Envs>>)
[[nodiscard]] _CCCL_HOST_API size_t
count(const _S& data, const _Envs& envs, view_element_t<_S> value, const _CallEnv& call_env = {})
{
  return sharded::count_if(data, envs, reserved::equals_value_fn<view_element_t<_S>>{value}, call_env);
}

/// @brief Count elements equal to @p value (generic, self-bound).
_CCCL_TEMPLATE(class _S, class _CallEnv = default_call_env)
_CCCL_REQUIRES(self_bound<::cuda::std::remove_cvref_t<_S>>)
[[nodiscard]] _CCCL_HOST_API size_t count(const _S& data, view_element_t<_S> value, const _CallEnv& call_env = {})
{
  const auto envs = default_envs(data);
  return sharded::count_if(data, envs, reserved::equals_value_fn<view_element_t<_S>>{value}, call_env);
}
} // namespace cuda::experimental::sharded
