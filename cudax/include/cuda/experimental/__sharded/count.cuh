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
#include <cuda/experimental/__sharded/sharded_array.cuh>

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

/**
 * @brief Count the elements satisfying a predicate.
 *
 * Phase 1 runs a CUB transform-reduce (predicate mapped to 0/1, summed) per
 * shard on the shard's stream, with temporaries allocated from the shard's
 * place; phase 2 sums the per-place counts. SYNCHRONOUS: returns the count.
 *
 * @param group the place group providing per-place memory resources
 * @param data  the sharded input (not modified)
 * @param pred  host- and device-callable predicate: `bool operator()(T)`
 */
template <typename _Tp, typename _Pred>
[[nodiscard]] _CCCL_HOST_API size_t count_if(place_group& group, const sharded_array<_Tp>& data, _Pred pred)
{
  if (data.empty())
  {
    return 0;
  }

  const size_t num_shards = data.num_shards();

  // Pinned host memory for the per-place counts (zero-initialized so skipped
  // empty shards contribute nothing)
  places::place_memory_resource host_mr(data_place::host());
  size_t* h_counts = static_cast<size_t*>(host_mr.allocate_sync(num_shards * sizeof(size_t), alignof(size_t)));
  ::std::fill(h_counts, h_counts + num_shards, size_t{0});

  // Phase 1: local transform-reduce on each shard; free the per-shard outputs
  // only after the final sync (places without stream-ordered deallocation)
  ::std::vector<::std::pair<places::place_memory_resource, size_t*>> d_outputs;
  d_outputs.reserve(num_shards);

  data.each_shard->*[&](const size_t g, const auto& s) {
    places::place_memory_resource mr(s.place);
    size_t* d_out = static_cast<size_t*>(mr.allocate(::cuda::stream_ref{s.stream}, sizeof(size_t), alignof(size_t)));
    d_outputs.emplace_back(mr, d_out);

    // Temporaries come from the shard's place through the group's resources
    const auto env = group.env(s.place, s.stream);
    cuda_safe_call(cub::DeviceReduce::TransformReduce(
      s.data,
      d_out,
      s.size,
      ::cuda::std::plus<size_t>{},
      reserved::count_transform_fn<_Tp, _Pred>{pred},
      size_t{0},
      env));

    cuda_safe_call(cudaMemcpyAsync(&h_counts[g], d_out, sizeof(size_t), cudaMemcpyDeviceToHost, s.stream));
  };

  data.sync();

  // Phase 2: sum the per-place counts
  size_t total = 0;
  for (size_t g = 0; g < num_shards; g++)
  {
    total += h_counts[g];
  }

  for (auto& [mr, ptr] : d_outputs)
  {
    mr.deallocate_sync(ptr, sizeof(size_t), alignof(size_t));
  }
  host_mr.deallocate_sync(h_counts, num_shards * sizeof(size_t), alignof(size_t));

  return total;
}

/// @brief Count the elements equal to `value`.
template <typename _Tp>
[[nodiscard]] _CCCL_HOST_API size_t count(place_group& group, const sharded_array<_Tp>& data, _Tp value)
{
  return count_if(group, data, reserved::equals_value_fn<_Tp>{value});
}
} // namespace cuda::experimental::sharded
