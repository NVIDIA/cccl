// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/device/device_segmented_scan.cuh>

#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/memory.h>
#include <thrust/scan.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/iterator>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/cmath>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/random>

#include <nvbench_helper.cuh>

#if !TUNE_BASE
template <typename AccumT>
struct load_balanced_policy_selector_t
{
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr auto operator()(cuda::compute_capability) const
    -> cub::SegmentedScanLoadBalancedPolicy
  {
    constexpr int pair_size               = int{sizeof(cub::KeyValuePair<int, AccumT>)};
    constexpr int min_tiles_per_block     = 16;
    constexpr int max_subscription_factor = 5;
    const auto scaled       = cub::detail::scale_mem_bound(1 << TUNE_THREADS_PER_BLOCK_POW2, TUNE_ITEMS, pair_size);
    const bool large_values = pair_size > 128;

    return cub::SegmentedScanLoadBalancedPolicy{
      scaled.threads_per_block,
      scaled.items_per_thread,
      large_values ? cub::BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED : cub::BLOCK_LOAD_WARP_TRANSPOSE,
      cub::LOAD_DEFAULT,
      large_values ? cub::BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED : cub::BLOCK_STORE_WARP_TRANSPOSE,
      cub::BLOCK_SCAN_WARP_SCANS,
      min_tiles_per_block,
      max_subscription_factor};
  }
};
#endif // !TUNE_BASE

enum class schedule
{
  default_,
  num_items,
  load_balancing
};

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  schedule,
  [](schedule value) {
    switch (value)
    {
      case schedule::default_:
        return "default";
      case schedule::num_items:
        return "num_items";
      case schedule::load_balancing:
        return "load_balancing";
    }
    return "unknown";
  },
  [](auto) {
    return std::string{};
  })

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

    const auto scaled_offset = static_cast<double>(elements) * cumulative_weights[index - 1] * inverse_weight_sum;
    return static_cast<OffsetT>(::cuda::std::floor(scaled_offset + 0.5));
  }
};

template <typename OffsetT>
[[nodiscard]] thrust::device_vector<OffsetT>
generate_pareto_segment_offsets(OffsetT elements, OffsetT num_segments, double alpha, seed_type shuffle_seed)
{
  const auto count = static_cast<::cuda::std::uint64_t>(num_segments);
  ::cuda::std::philox4x32 rng(shuffle_seed);
  const auto weights = thrust::make_transform_iterator(
    ::cuda::shuffle_iterator<::cuda::std::uint64_t>(count, rng), pareto_weight{count, alpha});

  auto cumulative_weights = thrust::device_vector<double>(num_segments, thrust::no_init);
  thrust::inclusive_scan(weights, weights + num_segments, cumulative_weights.begin());
  const double weight_sum = cumulative_weights.back();

  auto offsets = thrust::device_vector<OffsetT>(num_segments + 1, thrust::no_init);
  thrust::tabulate(offsets.begin(),
                   offsets.end(),
                   cumulative_to_offset<OffsetT>{
                     thrust::raw_pointer_cast(cumulative_weights.data()), 1.0 / weight_sum, elements, num_segments});

  return offsets;
}

template <typename OffsetT>
struct uniform_offset
{
  OffsetT elements;
  OffsetT segment_size;

  [[nodiscard]] _CCCL_HOST_DEVICE_API OffsetT operator()(OffsetT segment) const noexcept
  {
    return (::cuda::std::min) (elements, segment * segment_size);
  }
};

template <typename OffsetT>
[[nodiscard]] thrust::device_vector<OffsetT>
generate_uniform_segment_offsets(OffsetT elements, OffsetT num_segments, OffsetT segment_size)
{
  auto offsets = thrust::device_vector<OffsetT>(num_segments + 1, thrust::no_init);
  thrust::tabulate(offsets.begin(), offsets.end(), uniform_offset<OffsetT>{elements, segment_size});
  return offsets;
}

template <typename OffsetT>
struct add_gaps
{
  OffsetT gap;

  [[nodiscard]] _CCCL_HOST_DEVICE_API OffsetT operator()(OffsetT offset, OffsetT segment) const noexcept
  {
    return offset + segment * gap;
  }
};

// Gap{io} unused items follow every segment, in the input and in the output.
template <bool Uniform, typename T, typename OffsetT, schedule Schedule>
void variable_size_segments(nvbench::state& state, nvbench::type_list<T, OffsetT, nvbench::enum_type<Schedule>>)
{
  if (state.get_int64("Elements{io}") <= 0 || state.get_int64("MeanSegmentSize{io}") <= 0)
  {
    state.skip("Elements{io} and MeanSegmentSize{io} must be positive");
    return;
  }

  const auto elements          = static_cast<OffsetT>(state.get_int64("Elements{io}"));
  const auto mean_segment_size = static_cast<OffsetT>(state.get_int64("MeanSegmentSize{io}"));
  const auto gap_size          = state.get_int64("Gap{io}");
  const auto num_segments      = ::cuda::ceil_div(elements, mean_segment_size);

  const auto max_offset = static_cast<::cuda::std::int64_t>(::cuda::std::numeric_limits<OffsetT>::max());
  if (gap_size < 0
      || gap_size > (max_offset - static_cast<::cuda::std::int64_t>(elements))
                      / static_cast<::cuda::std::int64_t>(num_segments))
  {
    state.skip("Gap{io} must be non-negative and the gapped input must fit in OffsetT");
    return;
  }
  const auto input_elements =
    static_cast<::cuda::std::int64_t>(elements) + gap_size * static_cast<::cuda::std::int64_t>(num_segments);
  const auto gap = static_cast<OffsetT>(gap_size);

  auto& summary = state.add_summary("user/derived/segment_count");
  summary.set_string("name", "#Segments");
  summary.set_int64("value", num_segments);

  const thrust::device_vector<T> input = generate(static_cast<std::size_t>(input_elements));
  thrust::device_vector<T> output(static_cast<std::size_t>(input_elements), thrust::default_init);
  const auto offsets = [&] {
    if constexpr (Uniform)
    {
      return generate_uniform_segment_offsets(elements, num_segments, mean_segment_size);
    }
    else
    {
      return generate_pareto_segment_offsets(
        elements,
        num_segments,
        state.get_float64("Alpha{io}"),
        static_cast<seed_type>(state.get_int64("ShuffleSeed{io}")));
    }
  }();

  auto begin_offsets = thrust::device_vector<OffsetT>(gap == 0 ? OffsetT{0} : num_segments, thrust::no_init);
  auto end_offsets   = thrust::device_vector<OffsetT>(gap == 0 ? OffsetT{0} : num_segments, thrust::no_init);
  if (gap != 0)
  {
    thrust::transform(
      offsets.begin(),
      offsets.end() - 1,
      ::cuda::counting_iterator<OffsetT>{0},
      begin_offsets.begin(),
      add_gaps<OffsetT>{gap});
    thrust::transform(
      offsets.begin() + 1,
      offsets.end(),
      ::cuda::counting_iterator<OffsetT>{0},
      end_offsets.begin(),
      add_gaps<OffsetT>{gap});
  }

  const T* d_input         = thrust::raw_pointer_cast(input.data());
  T* d_output              = thrust::raw_pointer_cast(output.data());
  const OffsetT* d_offsets = thrust::raw_pointer_cast(offsets.data());
  const OffsetT* d_begins  = gap == 0 ? d_offsets : thrust::raw_pointer_cast(begin_offsets.data());
  const OffsetT* d_ends    = gap == 0 ? d_offsets + 1 : thrust::raw_pointer_cast(end_offsets.data());

  state.add_element_count(elements, "Elements");
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_reads<OffsetT>(gap == 0 ? num_segments + 1 : 2 * num_segments);
  state.add_global_memory_writes<T>(elements);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    const auto run = [&](const auto& env) {
      _CCCL_TRY_RUNTIME_API(
        cub::DeviceSegmentedScan::ExclusiveSegmentedScan,
        "ExclusiveSegmentedScan failed",
        d_input,
        d_output,
        d_begins,
        d_ends,
        d_begins,
        num_segments,
        op_t{},
        T{},
        env);
    };

    if constexpr (Schedule == schedule::default_)
    {
      const auto env = cub_bench_env(alloc, launch);
      run(env);
    }
    else if constexpr (Schedule == schedule::num_items)
    {
      const auto env = cub_bench_env(alloc, launch, cub::segmented_scan_num_items(elements));
      run(env);
    }
    else
    {
      const auto env = cub_bench_env(
        alloc,
        launch,
        cub::segmented_scan_load_balancing
#if !TUNE_BASE
        ,
        cuda::execution::tune(load_balanced_policy_selector_t<::cuda::std::__accumulator_t<op_t, T, T>>{})
#endif // !TUNE_BASE
      );
      run(env);
    }
  });
}

template <typename T, typename OffsetT, schedule Schedule>
void skewed_size_segments(nvbench::state& state, nvbench::type_list<T, OffsetT, nvbench::enum_type<Schedule>> types)
{
  variable_size_segments<false>(state, types);
}

template <typename T, typename OffsetT, schedule Schedule>
void uniform_size_segments(nvbench::state& state, nvbench::type_list<T, OffsetT, nvbench::enum_type<Schedule>> types)
{
  variable_size_segments<true>(state, types);
}

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

using schedules = nvbench::enum_type_list<schedule::default_, schedule::num_items, schedule::load_balancing>;

NVBENCH_BENCH_TYPES(skewed_size_segments, NVBENCH_TYPE_AXES(value_types, some_offset_types, schedules))
  .set_name("skewed_size_segments")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}", "Schedule{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", {22, 26})
  .add_int64_axis("MeanSegmentSize{io}", {32, 64, 128, 256, 512, 1024, 2048})
  .add_float64_axis("Alpha{io}", {2.0, 1.5, 1.3})
  .add_int64_axis("ShuffleSeed{io}", {42})
  .add_int64_axis("Gap{io}", {0});

NVBENCH_BENCH_TYPES(uniform_size_segments, NVBENCH_TYPE_AXES(value_types, some_offset_types, schedules))
  .set_name("uniform_size_segments")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}", "Schedule{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", {22, 26})
  .add_int64_axis("MeanSegmentSize{io}", {32, 64, 128, 256, 512, 1024, 2048})
  .add_int64_axis("Gap{io}", {0});
