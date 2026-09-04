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
 * @brief Equal-width histograms over sharded arrays: each place runs the
 *        device-scope primitive (CUB `DeviceHistogram::HistogramEven`) on its
 *        shard, then the per-place bin counts are summed — histogram bins are
 *        associative, so the cross-place combine is one addition per bin.
 *
 * Histogramming is read-only: it never mutates shard sizes, so it is
 * available on every sharded array, including contiguous
 * (`allocate_contiguous`) ones.
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

#include <cub/device/device_histogram.cuh>

#include <cuda/experimental/__places/place_group.cuh>
#include <cuda/experimental/__sharded/composition.cuh>
#include <cuda/experimental/__sharded/concepts.cuh>
#include <cuda/experimental/__sharded/cuda_safe_call.cuh>
#include <cuda/experimental/__sharded/default_envs.cuh>
#include <cuda/experimental/__sharded/pinned_staging.cuh>
#include <cuda/experimental/__sharded/sharded_array.cuh>
#include <cuda/experimental/__sharded/stream_scope.cuh>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace cuda::experimental::sharded
{
// ============================================================================
// Concept-generic tier: histogram over any sharded_view
// ============================================================================

/**
 * @brief Even-binned histogram over any `sharded_view`: per-shard
 * `cub::DeviceHistogram::HistogramEven` on each shard's environment, then a
 * per-bin host sum. SYNCHRONOUS-only (host combine): refuses at entry under
 * `sync_policy::forbid` and under capture. Staging via the call
 * environment's resource when present, the pinned arena otherwise.
 */
_CCCL_TEMPLATE(class _S, class _Envs, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  sharded_view<::cuda::std::remove_cvref_t<_S>> _CCCL_AND sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Envs>>)
[[nodiscard]] _CCCL_HOST_API ::std::vector<size_t> histogram_even(
  const _S& data,
  const _Envs& envs,
  int num_bins,
  view_element_t<_S> lower_level,
  view_element_t<_S> upper_level,
  const _CallEnv& call_env = {})
{
  if (num_bins <= 0)
  {
    _CCCL_THROW(::std::invalid_argument, "sharded::histogram_even: num_bins must be positive");
  }
  if (!(lower_level < upper_level))
  {
    _CCCL_THROW(::std::invalid_argument, "sharded::histogram_even: lower_level must be less than upper_level");
  }
  const size_t bins = static_cast<size_t>(num_bins);
  ::std::vector<size_t> counts(bins, 0);

  const ::std::size_t num_shards = reserved::__shard_count(data);
  if (reserved::__env_count(envs) < num_shards)
  {
    _CCCL_THROW(::std::invalid_argument, "sharded::histogram_even: fewer environments than shards");
  }
  if (num_shards == 0)
  {
    return counts;
  }

  // Refusals first, before any CUDA call: this form synchronizes.
  require_sync_allowed(call_env, "sharded::histogram_even (synchronous form)");
  places::check_not_capturing(nullptr, "sharded::histogram_even");
  for (const auto g : each(num_shards))
  {
    places::check_not_capturing(::cuda::get_stream(envs[g]).get(), "sharded::histogram_even");
  }

  using counter_type          = unsigned long long; // device-atomics-capable bin counter
  constexpr bool __env_has_mr = ::cuda::std::execution::__queryable_with<_CallEnv, ::cuda::mr::get_memory_resource_t>
                             || ::cuda::mr::__has_member_get_resource<_CallEnv>;
  const size_t host_bytes     = num_shards * bins * sizeof(counter_type);
  counter_type* h_hists       = nullptr;
  if constexpr (__env_has_mr)
  {
    auto staging_mr = ::cuda::mr::get_memory_resource(call_env);
    h_hists         = static_cast<counter_type*>(staging_mr.allocate_sync(host_bytes, alignof(counter_type)));
  }
  else
  {
    h_hists = static_cast<counter_type*>(reserved::__pinned_staging(host_bytes));
  }
  ::std::fill(h_hists, h_hists + num_shards * bins, counter_type{0});

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
    auto mr = ::cuda::mr::get_memory_resource(env);
    counter_type* d_hist =
      static_cast<counter_type*>(mr.allocate(shard_stream, bins * sizeof(counter_type), alignof(counter_type)));
    cuda_safe_call(
      cub::DeviceHistogram::HistogramEven(s.data, d_hist, num_bins + 1, lower_level, upper_level, s.size, env));
    cuda_safe_call(cudaMemcpyAsync(
      &h_hists[g * bins], d_hist, bins * sizeof(counter_type), cudaMemcpyDeviceToHost, shard_stream.get()));
    mr.deallocate(shard_stream, d_hist, bins * sizeof(counter_type), alignof(counter_type)); // after the copy
  }

  barrier(envs);

  for (const auto g : each(num_shards))
  {
    for (size_t b = 0; b < bins; b++)
    {
      counts[b] += static_cast<size_t>(h_hists[g * bins + b]);
    }
  }

  if constexpr (__env_has_mr)
  {
    auto staging_mr = ::cuda::mr::get_memory_resource(call_env);
    staging_mr.deallocate_sync(h_hists, host_bytes, alignof(counter_type));
  }
  // (arena staging is cached; nothing to release)

  return counts;
}

/// @brief Even-binned histogram (generic, self-bound).
_CCCL_TEMPLATE(class _S, class _CallEnv = default_call_env)
_CCCL_REQUIRES(self_bound<::cuda::std::remove_cvref_t<_S>>)
[[nodiscard]] _CCCL_HOST_API ::std::vector<size_t> histogram_even(
  const _S& data,
  int num_bins,
  view_element_t<_S> lower_level,
  view_element_t<_S> upper_level,
  const _CallEnv& call_env = {})
{
  const auto envs = default_envs(data);
  return sharded::histogram_even(data, envs, num_bins, lower_level, upper_level, call_env);
}
} // namespace cuda::experimental::sharded
