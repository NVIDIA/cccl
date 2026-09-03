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
 * @brief In-place deduplication of consecutive equal elements over sharded
 *        arrays (`std::unique` semantics): each place runs the device-scope
 *        primitive (CUB `DeviceSelect::Unique`, in place) on its shard, then a
 *        boundary pass trims duplicates that straddle shard boundaries — when
 *        a shard's last element equals the next non-empty shard's first, the
 *        earlier shard drops it (an O(1) size decrement, no data movement).
 *
 * `unique` MUTATES shard sizes and therefore refuses contiguous
 * (`allocate_contiguous`) arrays with `std::invalid_argument` — see the
 * rationale in `copy_if.cuh`.
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

#include <cub/device/device_select.cuh>

#include <cuda/std/cstdint>

#include <cuda/experimental/__places/place_group.cuh>
#include <cuda/experimental/__stf/utility/exception_policy.cuh> // SCOPE
#include <cuda/experimental/__sharded/concepts.cuh>
#include <cuda/experimental/__sharded/default_envs.cuh>
#include <cuda/experimental/__sharded/pinned_staging.cuh>
#include <cuda/experimental/__sharded/stream_scope.cuh>
#include <cuda/experimental/__sharded/sharded_array.cuh>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <cuda_runtime.h>

namespace cuda::experimental::sharded
{
namespace reserved
{
template <typename _Tp>
[[nodiscard]] _CCCL_HOST_API size_t unique_impl(place_group& group, sharded_array<_Tp>& data)
{
  if (data.empty())
  {
    return 0;
  }

  // Size write-back requires host synchronization: cannot be captured
  reserved::check_not_capturing(data, "sharded::unique");

  const size_t num_shards = data.num_shards();
  using count_type        = ::cuda::std::int64_t;

  // Pinned host staging: per-shard kept counts (zero-initialized so skipped
  // empty shards stay empty) and the post-unique boundary elements.
  // The _Tp block leads and the count block starts at an offset rounded up
  // to alignof(count_type); the arena block itself is cudaMallocHost-backed
  // (>= 256-byte aligned), so both blocks are aligned for their types.
  const size_t tp_bytes     = 2 * num_shards * sizeof(_Tp);
  const size_t count_offset = (tp_bytes + alignof(count_type) - 1) / alignof(count_type) * alignof(count_type);
  const size_t host_bytes   = count_offset + num_shards * sizeof(count_type);
  auto* h_base              = static_cast<unsigned char*>(reserved::__pinned_staging(host_bytes));
  _Tp* h_first                   = reinterpret_cast<_Tp*>(h_base);
  _Tp* h_last                    = h_first + num_shards;
  count_type* h_new_sizes        = reinterpret_cast<count_type*>(h_base + count_offset);
  ::std::fill(h_new_sizes, h_new_sizes + num_shards, count_type{0});

  // Phase 0: acquire every fallible resource BEFORE any shard is mutated, so
  // the likely failure (allocation) leaves the array untouched. The guard is
  // armed before the loop: a mid-loop failure frees what was acquired.
  ::std::vector<::std::tuple<places::place_memory_resource, count_type*, cudaStream_t>> d_counts;
  d_counts.reserve(num_shards);
  SCOPE(exit)
  {
    for (auto& [mr, ptr, stream] : d_counts)
    {
      mr.deallocate(::cuda::stream_ref{stream}, ptr, sizeof(count_type), alignof(count_type));
    }
  };
  data.each_shard->*[&](auto& s) {
    places::place_memory_resource mr(s.place);
    count_type* d_num =
      static_cast<count_type*>(mr.allocate(::cuda::stream_ref{s.stream}, sizeof(count_type), alignof(count_type)));
    d_counts.emplace_back(mv(mr), d_num, s.stream);
  };

  // Phase 1: local in-place unique on each shard (CUB keeps the first element
  // of every run of consecutive equal elements). From here through
  // commit_sizes (phases 1-3), the compaction is irrevocable: on any failure,
  // leave the array VALID and EMPTY rather than compacted with stale sizes.
  SCOPE(fail)
  {
    data.commit_sizes(::std::vector<size_t>(num_shards, 0));
  };
  size_t next = 0;
  data.each_shard->*[&](const size_t g, auto& s) {
    count_type* d_num = ::std::get<1>(d_counts[next++]);

    // Temporaries come from the shard's place through the group's resources
    const auto env = group.env(s.place, s.stream);
    cuda_safe_call(cub::DeviceSelect::Unique(s.data, d_num, static_cast<::cuda::std::int64_t>(s.size), env));

    cuda_safe_call(cudaMemcpyAsync(&h_new_sizes[g], d_num, sizeof(count_type), cudaMemcpyDeviceToHost, s.stream));
  };

  data.sync();

  for (size_t g = 0; g < num_shards; g++)
  {
    _CCCL_ASSERT(h_new_sizes[g] >= 0 && static_cast<size_t>(h_new_sizes[g]) <= data.shard(g).size,
                 "sharded::unique: select returned an out-of-range count");
  }

  // Phase 2: fetch the boundary elements of each locally deduplicated shard
  data.each_shard->*[&](const size_t g, auto& s) {
    const count_type n = h_new_sizes[g];
    if (n == 0)
    {
      return;
    }
    cuda_safe_call(cudaMemcpyAsync(&h_first[g], s.data, sizeof(_Tp), cudaMemcpyDeviceToHost, s.stream));
    cuda_safe_call(cudaMemcpyAsync(&h_last[g], s.data + n - 1, sizeof(_Tp), cudaMemcpyDeviceToHost, s.stream));
  };

  data.sync();

  // Phase 3: trim duplicates straddling shard boundaries. Whenever a shard's
  // last element equals the FIRST element of the next non-empty shard, the
  // earlier shard drops its last element (within a shard elements are already
  // consecutive-unique, so at most one element per boundary can match, and
  // the shard's new last element cannot re-match).
  size_t prev = num_shards; // index of the previous non-empty shard, if any
  for (size_t g = 0; g < num_shards; g++)
  {
    if (h_new_sizes[g] == 0)
    {
      continue;
    }
    if (prev != num_shards && h_last[prev] == h_first[g])
    {
      h_new_sizes[prev]--;
    }
    prev = g;
  }

  // Fold the new sizes into the container's bookkeeping through the owning
  // structure's atomic size-mutation verb
  ::std::vector<size_t> new_sizes(num_shards);
  for (size_t g = 0; g < num_shards; g++)
  {
    new_sizes[g] = static_cast<size_t>(h_new_sizes[g]);
  }
  data.commit_sizes(new_sizes);
  // (counters freed by SCOPE(exit); arena staging is cached)

  return data.size();
}
} // namespace reserved

/**
 * @brief Remove consecutive duplicate elements, in place (`std::unique`
 *        semantics over the logical array, across shard boundaries).
 *
 * Each shard is deduplicated locally (CUB `DeviceSelect::Unique`), then
 * boundary duplicates between consecutive non-empty shards are trimmed with
 * an O(1) size decrement per boundary. Shard sizes and global offsets are
 * updated; capacities are unchanged (`reset_sizes_to_capacity()` reuses the
 * buffers). SYNCHRONOUS: returns the total number of kept elements.
 *
 * @param group the place group providing per-place memory resources
 * @param data  the sharded array to deduplicate in place
 *
 * @throws std::invalid_argument on contiguous (`allocate_contiguous`) arrays:
 *         shrinking shard sizes would leave gaps between shards' valid
 *         elements, falsifying the read-as-one-array contract of
 *         `contiguous_data()`, and compacting across the gaps would migrate
 *         elements across the requested placement.
 */
template <typename _Tp>
[[nodiscard]] _CCCL_HOST_API size_t unique(place_group& group, sharded_array<_Tp>& data)
{
  if (data.is_contiguous())
  {
    _CCCL_THROW(::std::invalid_argument,
                "sharded::unique: not supported on contiguous (allocate_contiguous) arrays -- shrinking shard sizes "
                "would leave gaps between shards' valid elements, falsifying the read-as-one-array contract of "
                "contiguous_data(), and compacting across the gaps would migrate elements across the placement the "
                "caller asked for. Use a non-contiguous sharded_array, or copy into one first.");
  }
  return reserved::unique_impl(group, data);
}

// ============================================================================
// Concept-generic tier: unique over any owning_sharded structure
// ============================================================================

/**
 * @brief Remove consecutive duplicate elements in place (`std::unique`
 * semantics across the global index space) over any `owning_sharded`
 * structure: per-shard in-place `DeviceSelect::Unique` on each shard's
 * environment, then duplicates straddling shard boundaries are trimmed
 * (O(1) size decrement per boundary) and the sizes commit atomically.
 * Returns the new total element count.
 *
 * SYNCHRONOUS-only; refuses at entry under `sync_policy::forbid`, capture,
 * and on models whose sizes cannot be mutated (probed by committing the
 * current sizes before anything changes). Exception discipline as reviewed:
 * resources acquired before mutation; once the first select has run, any
 * failure leaves the structure VALID and EMPTY.
 */
_CCCL_TEMPLATE(class _S, class _Envs, class _CallEnv = default_call_env)
_CCCL_REQUIRES(owning_sharded<::cuda::std::remove_cvref_t<_S>> _CCCL_AND
                 sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Envs>>)
[[nodiscard]] _CCCL_HOST_API size_t unique(_S&& data, const _Envs& envs, const _CallEnv& call_env = {})
{
  using elem_t                   = view_element_t<_S>;
  using count_type               = ::cuda::std::int64_t;
  const ::std::size_t num_shards = static_cast<::std::size_t>(data.num_shards());
  if (static_cast<::std::size_t>(envs.size()) < num_shards)
  {
    _CCCL_THROW(::std::invalid_argument, "sharded::unique: fewer environments than shards");
  }
  if (num_shards == 0)
  {
    return 0;
  }

  // Refusals first, before any CUDA call: size write-back synchronizes.
  require_sync_allowed(call_env, "sharded::unique (synchronous form)");
  places::check_not_capturing(nullptr, "sharded::unique");
  for (::std::size_t g = 0; g < num_shards; g++)
  {
    places::check_not_capturing(::cuda::get_stream(envs[g]).get(), "sharded::unique");
  }

  // Entry probe: committing the current sizes throws exactly when this model
  // cannot mutate sizes, before any element moves.
  ::std::vector<size_t> new_sizes(num_shards);
  for (::std::size_t g = 0; g < num_shards; g++)
  {
    new_sizes[g] = static_cast<size_t>(data.shard(g).size);
  }
  data.commit_sizes(new_sizes);

  // Host staging: two boundary-element blocks (_Tp leads) + the count block
  // at an offset rounded to its alignment; the backing (arena or call-env
  // resource) is at least max_align-aligned.
  constexpr bool __env_has_mr =
    ::cuda::std::execution::__queryable_with<_CallEnv, ::cuda::mr::get_memory_resource_t>
    || ::cuda::mr::__has_member_get_resource<_CallEnv>;
  const size_t tp_bytes     = 2 * num_shards * sizeof(elem_t);
  const size_t count_offset = (tp_bytes + alignof(count_type) - 1) / alignof(count_type) * alignof(count_type);
  const size_t host_bytes   = count_offset + num_shards * sizeof(count_type);
  unsigned char* h_base     = nullptr;
  if constexpr (__env_has_mr)
  {
    auto staging_mr = ::cuda::mr::get_memory_resource(call_env);
    h_base          = static_cast<unsigned char*>(staging_mr.allocate_sync(host_bytes, alignof(::std::max_align_t)));
  }
  else
  {
    h_base = static_cast<unsigned char*>(reserved::__pinned_staging(host_bytes));
  }
  elem_t* h_first         = reinterpret_cast<elem_t*>(h_base);
  elem_t* h_last          = h_first + num_shards;
  count_type* h_new_sizes = reinterpret_cast<count_type*>(h_base + count_offset);
  ::std::fill(h_new_sizes, h_new_sizes + num_shards, count_type{0});

  // Phase 0: acquire every fallible resource before any shard is mutated.
  using scratch_mr_t = ::cuda::std::remove_cvref_t<decltype(::cuda::mr::get_memory_resource(envs[::std::size_t{0}]))>;
  ::std::vector<::std::tuple<scratch_mr_t, count_type*, cudaStream_t>> d_counts;
  d_counts.reserve(num_shards);
  SCOPE(exit)
  {
    for (auto& [mr, ptr, stream] : d_counts)
    {
      mr.deallocate(::cuda::stream_ref{stream}, ptr, sizeof(count_type), alignof(count_type));
    }
  };
  for (::std::size_t g = 0; g < num_shards; g++)
  {
    if (data.shard(g).size == 0)
    {
      continue;
    }
    const ::cuda::stream_ref shard_stream = ::cuda::get_stream(envs[g]);
    auto mr                               = ::cuda::mr::get_memory_resource(envs[g]);
    count_type* d_num = static_cast<count_type*>(mr.allocate(shard_stream, sizeof(count_type), alignof(count_type)));
    d_counts.emplace_back(mv(mr), d_num, shard_stream.get());
  }

  // Phase 1: in-place unique per shard; irrevocable from here (valid-and-
  // empty on failure).
  SCOPE(fail)
  {
    data.commit_sizes(::std::vector<size_t>(num_shards, 0));
  };
  ::std::size_t next = 0;
  for (::std::size_t g = 0; g < num_shards; g++)
  {
    const auto& s = data.shard(g);
    if (s.size == 0)
    {
      continue;
    }
    count_type* d_num                     = ::std::get<1>(d_counts[next++]);
    const ::cuda::stream_ref shard_stream = ::cuda::get_stream(envs[g]);
    stream_scope scope(shard_stream.get());
    cuda_safe_call(cub::DeviceSelect::Unique(s.data, d_num, static_cast<::cuda::std::int64_t>(s.size), envs[g]));
    cuda_safe_call(
      cudaMemcpyAsync(&h_new_sizes[g], d_num, sizeof(count_type), cudaMemcpyDeviceToHost, shard_stream.get()));
  }
  for (::std::size_t g = 0; g < num_shards; g++)
  {
    if (data.shard(g).size != 0)
    {
      cuda_safe_call(cudaStreamSynchronize(::cuda::get_stream(envs[g]).get()));
    }
  }

  // Phase 2: fetch each locally deduplicated shard's boundary elements.
  for (::std::size_t g = 0; g < num_shards; g++)
  {
    const auto& s      = data.shard(g);
    const count_type n = h_new_sizes[g];
    if (n == 0)
    {
      continue;
    }
    const ::cuda::stream_ref shard_stream = ::cuda::get_stream(envs[g]);
    cuda_safe_call(cudaMemcpyAsync(&h_first[g], s.data, sizeof(elem_t), cudaMemcpyDeviceToHost, shard_stream.get()));
    cuda_safe_call(
      cudaMemcpyAsync(&h_last[g], s.data + n - 1, sizeof(elem_t), cudaMemcpyDeviceToHost, shard_stream.get()));
  }
  for (::std::size_t g = 0; g < num_shards; g++)
  {
    if (h_new_sizes[g] != 0)
    {
      cuda_safe_call(cudaStreamSynchronize(::cuda::get_stream(envs[g]).get()));
    }
  }

  // Phase 3: trim duplicates straddling shard boundaries (a shard drops its
  // last element when it equals the FIRST element of the next non-empty
  // shard; within a shard elements are already consecutive-unique, so at
  // most one element per boundary can match).
  ::std::size_t prev = num_shards;
  for (::std::size_t g = 0; g < num_shards; g++)
  {
    if (h_new_sizes[g] == 0)
    {
      continue;
    }
    if (prev != num_shards && h_last[prev] == h_first[g])
    {
      h_new_sizes[prev]--;
    }
    prev = g;
  }

  size_t total = 0;
  for (::std::size_t g = 0; g < num_shards; g++)
  {
    new_sizes[g] = static_cast<size_t>(h_new_sizes[g]);
    total += new_sizes[g];
  }
  data.commit_sizes(new_sizes);

  if constexpr (__env_has_mr)
  {
    auto staging_mr = ::cuda::mr::get_memory_resource(call_env);
    staging_mr.deallocate_sync(h_base, host_bytes, alignof(::std::max_align_t));
  }
  // (counters freed by SCOPE(exit); arena staging is cached)

  return total;
}

/// @brief Remove consecutive duplicates (generic, self-bound).
_CCCL_TEMPLATE(class _S, class _CallEnv = default_call_env)
_CCCL_REQUIRES(owning_sharded<::cuda::std::remove_cvref_t<_S>> _CCCL_AND
                 self_bound<::cuda::std::remove_cvref_t<_S>> _CCCL_AND(
                   !sharded_alloc_env_range<::cuda::std::remove_cvref_t<_CallEnv>>))
[[nodiscard]] _CCCL_HOST_API size_t unique(_S&& data, const _CallEnv& call_env = {})
{
  const auto envs = default_envs(data);
  return sharded::unique(::cuda::std::forward<_S>(data), envs, call_env);
}
} // namespace cuda::experimental::sharded
