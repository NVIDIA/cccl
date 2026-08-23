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
 * @brief In-place selection over sharded arrays (`copy_if` / `filter` /
 *        `remove_if`): each place runs the device-scope primitive (CUB
 *        `DeviceSelect::If`, in place) on its shard, then the container's
 *        shard sizes and offsets are updated to the compacted result.
 *
 * These algorithms MUTATE shard sizes. That is exactly what the contiguous
 * backing cannot represent: shrinking a shard would leave a gap between its
 * valid elements and the next shard's, falsifying the read-as-one-array
 * contract of `contiguous_data()`, while compacting across the gap would
 * migrate elements onto other places than the caller asked for. They
 * therefore refuse contiguous (`allocate_contiguous`) arrays with
 * `std::invalid_argument`.
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
/// @brief Negation of a predicate (for `remove_if`).
template <typename _Tp, typename _Pred>
struct negate_pred_fn
{
  _Pred pred;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE bool operator()(_Tp val) const
  {
    return !pred(val);
  }
};

template <typename _Tp, typename _Pred>
[[nodiscard]] _CCCL_HOST_API size_t copy_if_impl(place_group& group, sharded_array<_Tp>& data, _Pred pred)
{
  if (data.empty())
  {
    return 0;
  }

  // Size write-back requires host synchronization: cannot be captured
  reserved::check_not_capturing(data, "sharded::copy_if");

  const size_t num_shards = data.num_shards();
  using count_type        = ::cuda::std::int64_t;

  // Pinned host memory for the per-shard kept counts (zero-initialized so
  // skipped empty shards stay empty)
  places::place_memory_resource host_mr(data_place::host());
  count_type* h_new_sizes =
    static_cast<count_type*>(host_mr.allocate_sync(num_shards * sizeof(count_type), alignof(count_type)));
  ::std::fill(h_new_sizes, h_new_sizes + num_shards, count_type{0});

  // Phase 1: local in-place select on each shard (CUB compacts the kept
  // elements to the front of the shard, preserving their order); the
  // per-shard counters are freed only after the final sync
  ::std::vector<::std::pair<places::place_memory_resource, count_type*>> d_counts;
  d_counts.reserve(num_shards);

  data.each_shard->*[&](const size_t g, auto& s) {
    places::place_memory_resource mr(s.place);
    count_type* d_num =
      static_cast<count_type*>(mr.allocate(::cuda::stream_ref{s.stream}, sizeof(count_type), alignof(count_type)));
    d_counts.emplace_back(mr, d_num);

    // Temporaries come from the shard's place through the group's resources
    const auto env = group.env(s.place, s.stream);
    cuda_safe_call(cub::DeviceSelect::If(s.data, d_num, static_cast<::cuda::std::int64_t>(s.size), pred, env));

    cuda_safe_call(cudaMemcpyAsync(&h_new_sizes[g], d_num, sizeof(count_type), cudaMemcpyDeviceToHost, s.stream));
  };

  data.sync();

  // Phase 2: fold the new sizes into the container's bookkeeping
  for (size_t g = 0; g < num_shards; g++)
  {
    _CCCL_ASSERT(h_new_sizes[g] >= 0 && static_cast<size_t>(h_new_sizes[g]) <= data.shard(g).size,
                 "sharded::copy_if: select returned an out-of-range count");
    data.shard(g).size = static_cast<size_t>(h_new_sizes[g]);
  }
  data.recalculate_offsets();

  for (auto& [mr, ptr] : d_counts)
  {
    mr.deallocate_sync(ptr, sizeof(count_type), alignof(count_type));
  }
  host_mr.deallocate_sync(h_new_sizes, num_shards * sizeof(count_type), alignof(count_type));

  return data.size();
}
} // namespace reserved

/**
 * @brief Keep only the elements satisfying a predicate, in place.
 *
 * Each shard is compacted locally (order preserved); shard sizes and global
 * offsets are updated, capacities are unchanged (`reset_sizes_to_capacity()`
 * reuses the buffers). SYNCHRONOUS: returns the total number of kept
 * elements.
 *
 * @param group the place group providing per-place memory resources
 * @param data  the sharded array to filter in place
 * @param pred  host- and device-callable predicate: `bool operator()(T)`
 *
 * @throws std::invalid_argument on contiguous (`allocate_contiguous`) arrays:
 *         shrinking shard sizes would leave gaps between shards' valid
 *         elements, falsifying the read-as-one-array contract of
 *         `contiguous_data()`, and compacting across the gaps would migrate
 *         elements across the requested placement.
 */
template <typename _Tp, typename _Pred>
[[nodiscard]] _CCCL_HOST_API size_t copy_if(place_group& group, sharded_array<_Tp>& data, _Pred pred)
{
  if (data.is_contiguous())
  {
    _CCCL_THROW(::std::invalid_argument,
                "sharded::copy_if: not supported on contiguous (allocate_contiguous) arrays -- shrinking shard sizes "
                "would leave gaps between shards' valid elements, falsifying the read-as-one-array contract of "
                "contiguous_data(), and compacting across the gaps would migrate elements across the placement the "
                "caller asked for. Use a non-contiguous sharded_array, or copy into one first.");
  }
  return reserved::copy_if_impl(group, data, pred);
}

/// @brief Alias for `copy_if`.
template <typename _Tp, typename _Pred>
[[nodiscard]] _CCCL_HOST_API size_t filter(place_group& group, sharded_array<_Tp>& data, _Pred pred)
{
  return copy_if(group, data, pred);
}

/**
 * @brief Remove the elements satisfying a predicate, in place (the inverse of
 *        `copy_if`). SYNCHRONOUS: returns the total number of kept elements.
 *
 * @throws std::invalid_argument on contiguous arrays (see `copy_if`)
 */
template <typename _Tp, typename _Pred>
[[nodiscard]] _CCCL_HOST_API size_t remove_if(place_group& group, sharded_array<_Tp>& data, _Pred pred)
{
  if (data.is_contiguous())
  {
    _CCCL_THROW(::std::invalid_argument,
                "sharded::remove_if: not supported on contiguous (allocate_contiguous) arrays -- shrinking shard "
                "sizes would leave gaps between shards' valid elements, falsifying the read-as-one-array contract of "
                "contiguous_data(), and compacting across the gaps would migrate elements across the placement the "
                "caller asked for. Use a non-contiguous sharded_array, or copy into one first.");
  }
  return reserved::copy_if_impl(group, data, reserved::negate_pred_fn<_Tp, _Pred>{pred});
}
} // namespace cuda::experimental::sharded
