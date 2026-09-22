// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/device/device_segmented_scan.cuh>

#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/memory.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/shuffle.h>
#include <thrust/tabulate.h>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/std/cmath>
#include <cuda/std/cstdint>
#include <cuda/std/random>

#include <nvbench_helper.cuh>

namespace
{
using seed_type = ::cuda::std::philox4x32::result_type;

struct pareto_weight
{
  ::cuda::std::uint64_t count;
  double alpha;

  [[nodiscard]] _CCCL_HOST_DEVICE_API double operator()(::cuda::std::uint64_t index) const noexcept
  {
    const auto probability = (static_cast<double>(index) + 0.5) / static_cast<double>(count);
    return ::cuda::std::pow(1.0 - probability, -1.0 / alpha);
  }
};

template <typename OffsetT>
struct cumulative_to_offset
{
  const double* cumulative_weights;
  double inverse_weight_sum;
  OffsetT elements;
  OffsetT num_segments;

  [[nodiscard]] _CCCL_HOST_DEVICE_API OffsetT operator()(OffsetT index) const noexcept
  {
    if (index == 0)
    {
      return 0;
    }
    if (index == num_segments)
    {
      return elements;
    }

    const auto scaled_offset = static_cast<double>(elements) * cumulative_weights[index] * inverse_weight_sum;
    return static_cast<OffsetT>(::cuda::std::floor(scaled_offset + 0.5));
  }
};

template <typename OffsetT>
[[nodiscard]] thrust::device_vector<OffsetT>
generate_pareto_segment_offsets(OffsetT elements, OffsetT num_segments, double alpha, seed_type shuffle_seed)
{
  auto cumulative_weights = thrust::device_vector<double>(num_segments + 1, thrust::no_init);
  const auto weights       = thrust::make_transform_iterator(
    thrust::make_counting_iterator(::cuda::std::uint64_t{0}),
    pareto_weight{static_cast<::cuda::std::uint64_t>(num_segments), alpha});
  ::cuda::std::philox4x32 rng(shuffle_seed);
  thrust::shuffle_copy(weights, weights + num_segments, cumulative_weights.begin(), rng);

  const auto weight_sum = thrust::reduce(cumulative_weights.begin(), cumulative_weights.end() - 1, 0.0);
  thrust::exclusive_scan(cumulative_weights.begin(), cumulative_weights.end() - 1, cumulative_weights.begin());
  thrust::fill_n(cumulative_weights.end() - 1, 1, weight_sum);

  auto offsets = thrust::device_vector<OffsetT>(num_segments + 1, thrust::no_init);
  thrust::tabulate(
    offsets.begin(),
    offsets.end(),
    cumulative_to_offset<OffsetT>{
      thrust::raw_pointer_cast(cumulative_weights.data()),
      1.0 / weight_sum,
      elements,
      num_segments});

  return offsets;
}

template <typename T, typename OffsetT>
void skewed_size_segments(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  const auto elements          = static_cast<OffsetT>(state.get_int64("Elements{io}"));
  const auto mean_segment_size = static_cast<OffsetT>(state.get_int64("MeanSegmentSize{io}"));
  const auto alpha             = state.get_float64("Alpha{io}");
  const auto shuffle_seed      = static_cast<seed_type>(state.get_int64("ShuffleSeed{io}"));
  const auto num_segments      = ::cuda::ceil_div(elements, mean_segment_size);

  auto& summary = state.add_summary("user/derived/segment_count");
  summary.set_string("name", "#Segments");
  summary.set_int64("value", num_segments);

  const thrust::device_vector<T> input = generate(elements);
  thrust::device_vector<T> output(elements, thrust::default_init);
  const auto offsets = generate_pareto_segment_offsets(elements, num_segments, alpha, shuffle_seed);

  const T* d_input         = thrust::raw_pointer_cast(input.data());
  T* d_output              = thrust::raw_pointer_cast(output.data());
  const OffsetT* d_offsets = thrust::raw_pointer_cast(offsets.data());

  state.add_element_count(elements, "Elements");
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_reads<OffsetT>(num_segments + 1);
  state.add_global_memory_writes<T>(elements);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    auto env = cub_bench_env(alloc, launch);
    _CCCL_TRY_RUNTIME_API(
      cub::DeviceSegmentedScan::ExclusiveSegmentedScan,
      "ExclusiveSegmentedScan failed",
      d_input,
      d_output,
      d_offsets,
      d_offsets + 1,
      d_offsets,
      num_segments,
      op_t{},
      T{},
      env);
  });
}
} // namespace

#ifdef TUNE_T
using value_types = nvbench::type_list<TUNE_T>;
#else
using value_types = nvbench::type_list<int32_t, int64_t, float, double>;
#endif

#ifdef TUNE_OffsetT
using some_offset_types = nvbench::type_list<TUNE_OffsetT>;
#else
using some_offset_types = nvbench::type_list<int32_t>;
#endif

NVBENCH_BENCH_TYPES(skewed_size_segments, NVBENCH_TYPE_AXES(value_types, some_offset_types))
  .set_name("skewed_size_segments")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", {22, 26})
  .add_int64_axis("MeanSegmentSize{io}", {32, 64, 128, 256, 512, 1024, 2048})
  .add_float64_axis("Alpha{io}", {2.5, 2.0, 1.75, 1.5, 1.3})
  .add_int64_axis("ShuffleSeed{io}", {42});
