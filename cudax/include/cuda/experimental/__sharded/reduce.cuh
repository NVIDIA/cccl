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
 * @brief Reduction over sharded arrays: each place runs the device-scope
 *        primitive (CUB `DeviceReduce`) on its shard, then the per-place
 *        partials are combined — the same local-primitive-plus-combine
 *        structure the device scope itself uses over blocks.
 *
 * Algorithm temporaries are drawn from each shard's own place through the
 * group's per-place memory resources, so scratch lands where the work runs.
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

#include <cuda/functional>
#include <cuda/std/functional>
#include <cuda/std/limits>

#include <cuda/experimental/__places/place_group.cuh>
#include <cuda/experimental/__sharded/sharded_array.cuh>

#include <algorithm>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

namespace cuda::experimental::sharded
{
/**
 * @brief Reduce all elements with a custom operator.
 *
 * Phase 1 runs CUB `DeviceReduce` per shard on the shard's stream, with
 * temporaries allocated from the shard's place; phase 2 combines the
 * per-place partials. SYNCHRONOUS: returns the final value.
 *
 * @param group   the place group providing per-place memory resources
 * @param data    the sharded input (not modified)
 * @param reduce_op host- and device-callable binary operator
 * @param init_value initial (identity) value
 */
template <typename _Tp, typename _ReduceOp>
[[nodiscard]] _CCCL_HOST_API _Tp
reduce(place_group& group, const sharded_array<_Tp>& data, _ReduceOp reduce_op, _Tp init_value = _Tp{})
{
  if (data.empty())
  {
    return init_value;
  }

  // Host-side combine + synchronization: cannot be recorded into a CUDA graph
  reserved::check_not_capturing(data, "sharded::reduce");

  const size_t num_shards = data.num_shards();

  // Pinned host memory for the per-place partials (initialized so skipped
  // empty shards contribute the identity)
  places::place_memory_resource host_mr(data_place::host());
  _Tp* h_partials = static_cast<_Tp*>(host_mr.allocate_sync(num_shards * sizeof(_Tp), alignof(_Tp)));
  ::std::fill(h_partials, h_partials + num_shards, init_value);

  // Phase 1: local reduce on each shard; free the per-shard outputs only
  // after the final sync (places without stream-ordered deallocation)
  ::std::vector<::std::pair<places::place_memory_resource, _Tp*>> d_outputs;
  d_outputs.reserve(num_shards);

  data.each_shard->*[&](const size_t g, const auto& s) {
    places::place_memory_resource mr(s.place);
    _Tp* d_out = static_cast<_Tp*>(mr.allocate(::cuda::stream_ref{s.stream}, sizeof(_Tp), alignof(_Tp)));
    d_outputs.emplace_back(mr, d_out);

    // Temporaries come from the shard's place through the group's resources
    const auto env = group.env(s.place, s.stream);
    cuda_safe_call(cub::DeviceReduce::Reduce(s.data, d_out, s.size, reduce_op, init_value, env));

    cuda_safe_call(cudaMemcpyAsync(&h_partials[g], d_out, sizeof(_Tp), cudaMemcpyDeviceToHost, s.stream));
  };

  data.sync();

  // Phase 2: combine the per-place partials
  _Tp result = init_value;
  for (size_t g = 0; g < num_shards; g++)
  {
    result = reduce_op(result, h_partials[g]);
  }

  for (auto& [mr, ptr] : d_outputs)
  {
    mr.deallocate_sync(ptr, sizeof(_Tp), alignof(_Tp));
  }
  host_mr.deallocate_sync(h_partials, num_shards * sizeof(_Tp), alignof(_Tp));

  return result;
}

/// @brief Sum of all elements.
template <typename _Tp>
[[nodiscard]] _CCCL_HOST_API _Tp sum(place_group& group, const sharded_array<_Tp>& data)
{
  return reduce(group, data, ::cuda::std::plus<_Tp>{}, _Tp{0});
}

/// @brief Minimum element.
template <typename _Tp>
[[nodiscard]] _CCCL_HOST_API _Tp min(place_group& group, const sharded_array<_Tp>& data)
{
  return reduce(group, data, ::cuda::minimum<_Tp>{}, ::cuda::std::numeric_limits<_Tp>::max());
}

/// @brief Maximum element.
template <typename _Tp>
[[nodiscard]] _CCCL_HOST_API _Tp max(place_group& group, const sharded_array<_Tp>& data)
{
  return reduce(group, data, ::cuda::maximum<_Tp>{}, ::cuda::std::numeric_limits<_Tp>::lowest());
}
} // namespace cuda::experimental::sharded
