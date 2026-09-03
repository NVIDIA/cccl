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
#include <cuda/experimental/__sharded/concepts.cuh>
#include <cuda/experimental/__sharded/default_envs.cuh>
#include <cuda/experimental/__sharded/pinned_staging.cuh>
#include <cuda/experimental/__sharded/stream_scope.cuh>
#include <cuda/experimental/__sharded/sharded_array.cuh>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace cuda::experimental::sharded
{
/**
 * @brief Histogram with `num_bins` equal-width bins spanning
 *        `[lower_level, upper_level)`.
 *
 * Bin `b` counts the samples in
 * `[lower + b * (upper - lower) / num_bins, lower + (b+1) * (upper - lower) / num_bins)`;
 * samples outside `[lower_level, upper_level)` are ignored. Phase 1 runs CUB
 * `DeviceHistogram::HistogramEven` per shard on the shard's stream, with the
 * per-place histogram and temporaries allocated from the shard's place;
 * phase 2 sums the per-place bin counts. SYNCHRONOUS: returns the combined
 * histogram.
 *
 * @param group       the place group providing per-place memory resources
 * @param data        the sharded input (not modified)
 * @param num_bins    number of equal-width bins (>= 1)
 * @param lower_level inclusive lower bound of the lowest bin
 * @param upper_level exclusive upper bound of the highest bin
 *
 * @throws std::invalid_argument when `num_bins < 1` or
 *         `lower_level >= upper_level`
 */
template <typename _Tp, typename _LevelT>
[[nodiscard]] _CCCL_HOST_API ::std::vector<size_t> histogram_even(
  place_group& group, const sharded_array<_Tp>& data, int num_bins, _LevelT lower_level, _LevelT upper_level)
{
  if (num_bins < 1)
  {
    _CCCL_THROW(::std::invalid_argument,
                "sharded::histogram_even: num_bins (" + ::std::to_string(num_bins) + ") must be at least 1");
  }
  if (!(lower_level < upper_level))
  {
    _CCCL_THROW(::std::invalid_argument, "sharded::histogram_even: lower_level must be less than upper_level");
  }

  ::std::vector<size_t> counts(static_cast<size_t>(num_bins), 0);
  if (data.empty())
  {
    return counts;
  }

  // Host-side combine + synchronization: cannot be recorded into a CUDA graph
  reserved::check_not_capturing(data, "sharded::histogram_even");

  const size_t num_shards = data.num_shards();
  using counter_type      = unsigned long long; // device-atomics-capable bin counter
  const size_t bins       = static_cast<size_t>(num_bins);

  // Pinned host memory for the per-place histograms (zero-initialized so
  // skipped empty shards contribute nothing)
  const size_t host_bytes = num_shards * bins * sizeof(counter_type);
  counter_type* h_hists   = static_cast<counter_type*>(reserved::__pinned_staging(host_bytes));
  ::std::fill(h_hists, h_hists + num_shards * bins, counter_type{0});

  // Phase 1: local histogram on each shard; free the per-shard histograms
  // only after the final sync (places without stream-ordered deallocation)
  ::std::vector<::std::pair<places::place_memory_resource, counter_type*>> d_hists;
  d_hists.reserve(num_shards);

  data.each_shard->*[&](const size_t g, const auto& s) {
    places::place_memory_resource mr(s.place);
    counter_type* d_hist = static_cast<counter_type*>(
      mr.allocate(::cuda::stream_ref{s.stream}, bins * sizeof(counter_type), alignof(counter_type)));
    d_hists.emplace_back(mr, d_hist);

    // Temporaries come from the shard's place through the group's resources
    const auto env = group.env(s.place, s.stream);
    cuda_safe_call(
      cub::DeviceHistogram::HistogramEven(s.data, d_hist, num_bins + 1, lower_level, upper_level, s.size, env));

    cuda_safe_call(
      cudaMemcpyAsync(&h_hists[g * bins], d_hist, bins * sizeof(counter_type), cudaMemcpyDeviceToHost, s.stream));
  };

  data.sync();

  // Phase 2: sum the per-place bin counts
  for (size_t g = 0; g < num_shards; g++)
  {
    for (size_t b = 0; b < bins; b++)
    {
      counts[b] += static_cast<size_t>(h_hists[g * bins + b]);
    }
  }

  for (auto& [mr, ptr] : d_hists)
  {
    mr.deallocate_sync(ptr, bins * sizeof(counter_type), alignof(counter_type));
  }
  // (arena staging is cached; nothing to release)

  return counts;
}

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
_CCCL_REQUIRES(sharded_view<::cuda::std::remove_cvref_t<_S>> _CCCL_AND
                 sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Envs>>)
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
  const size_t bins = static_cast<size_t>(num_bins);
  ::std::vector<size_t> counts(bins, 0);

  const ::std::size_t num_shards = static_cast<::std::size_t>(data.num_shards());
  if (static_cast<::std::size_t>(envs.size()) < num_shards)
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
  for (::std::size_t g = 0; g < num_shards; g++)
  {
    places::check_not_capturing(::cuda::get_stream(envs[g]).get(), "sharded::histogram_even");
  }

  using counter_type = unsigned long long; // device-atomics-capable bin counter
  constexpr bool __env_has_mr =
    ::cuda::std::execution::__queryable_with<_CallEnv, ::cuda::mr::get_memory_resource_t>
    || ::cuda::mr::__has_member_get_resource<_CallEnv>;
  const size_t host_bytes = num_shards * bins * sizeof(counter_type);
  counter_type* h_hists   = nullptr;
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

  for (::std::size_t g = 0; g < num_shards; g++)
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

  for (::std::size_t g = 0; g < num_shards; g++)
  {
    if (data.shard(g).size != 0)
    {
      cuda_safe_call(cudaStreamSynchronize(::cuda::get_stream(envs[g]).get()));
    }
  }

  for (::std::size_t g = 0; g < num_shards; g++)
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
