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
#include <cuda/experimental/__sharded/sharded_array.cuh>

#include <algorithm>
#include <stdexcept>
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
  // empty shards stay empty) and the post-unique boundary elements
  places::place_memory_resource host_mr(data_place::host());
  // The _Tp block leads so both blocks are aligned for their types (the
  // count block only needs alignof(count_type) <= alignof(_Tp) or the
  // natural alignment of the offset, both guaranteed by the max() below).
  constexpr size_t staging_align = ::std::max(alignof(_Tp), alignof(count_type));
  const size_t tp_bytes          = 2 * num_shards * sizeof(_Tp);
  const size_t count_offset      = (tp_bytes + alignof(count_type) - 1) / alignof(count_type) * alignof(count_type);
  const size_t host_bytes        = count_offset + num_shards * sizeof(count_type);
  auto* h_base                   = static_cast<unsigned char*>(host_mr.allocate_sync(host_bytes, staging_align));
  _Tp* h_first                   = reinterpret_cast<_Tp*>(h_base);
  _Tp* h_last                    = h_first + num_shards;
  count_type* h_new_sizes        = reinterpret_cast<count_type*>(h_base + count_offset);
  ::std::fill(h_new_sizes, h_new_sizes + num_shards, count_type{0});

  // Phase 1: local in-place unique on each shard (CUB keeps the first element
  // of every run of consecutive equal elements); the per-shard counters are
  // freed only after the final sync
  ::std::vector<::std::pair<places::place_memory_resource, count_type*>> d_counts;
  d_counts.reserve(num_shards);

  data.each_shard->*[&](const size_t g, auto& s) {
    places::place_memory_resource mr(s.place);
    count_type* d_num =
      static_cast<count_type*>(mr.allocate(::cuda::stream_ref{s.stream}, sizeof(count_type), alignof(count_type)));
    d_counts.emplace_back(mr, d_num);

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

  // Fold the new sizes into the container's bookkeeping
  for (size_t g = 0; g < num_shards; g++)
  {
    data.shard(g).size = static_cast<size_t>(h_new_sizes[g]);
  }
  data.recalculate_offsets();

  for (auto& [mr, ptr] : d_counts)
  {
    mr.deallocate_sync(ptr, sizeof(count_type), alignof(count_type));
  }
  host_mr.deallocate_sync(h_base, host_bytes, staging_align);

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
} // namespace cuda::experimental::sharded
