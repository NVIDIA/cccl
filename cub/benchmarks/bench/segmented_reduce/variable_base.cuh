// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/device/dispatch/dispatch_segmented_reduce.cuh>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cuda/__execution/max_segment_size.h>
#include <cuda/std/type_traits>

#include <nvbench_helper.cuh>

// %RANGE% TUNE_ITEMS_PER_VEC_LOAD_POW2 ipv 1:2:1
// %RANGE% TUNE_S_THREADS_PER_WARP stpw 1:32:1
// %RANGE% TUNE_M_THREADS_PER_WARP mtpw 1:32:1
// %RANGE% TUNE_L_NOMINAL_4B_THREADS_PER_BLOCK ltpb 128:1024:32
// %RANGE% TUNE_S_NOMINAL_4B_ITEMS_PER_THREAD sipt 1:32:1
// %RANGE% TUNE_M_NOMINAL_4B_ITEMS_PER_THREAD mipt 1:32:1
// %RANGE% TUNE_L_NOMINAL_4B_ITEMS_PER_THREAD lipt 7:24:1

#ifndef TUNE_BASE
#  define TUNE_ITEMS_PER_VEC_LOAD (1 << TUNE_ITEMS_PER_VEC_LOAD_POW2)
#endif

#if !TUNE_BASE
template <typename AccumT>
struct policy_hub_t
{
  struct policy_t : cub::ChainedPolicy<300, policy_t, policy_t>
  {
    static constexpr int items_per_vec_load = TUNE_ITEMS_PER_VEC_LOAD;

    static constexpr int small_threads_per_warp  = TUNE_S_THREADS_PER_WARP;
    static constexpr int medium_threads_per_warp = TUNE_M_THREADS_PER_WARP;

    static constexpr int nominal_4b_large_threads_per_block = TUNE_L_NOMINAL_4B_THREADS_PER_BLOCK;

    static constexpr int nominal_4b_small_items_per_thread  = TUNE_S_NOMINAL_4B_ITEMS_PER_THREAD;
    static constexpr int nominal_4b_medium_items_per_thread = TUNE_M_NOMINAL_4B_ITEMS_PER_THREAD;
    static constexpr int nominal_4b_large_items_per_thread  = TUNE_L_NOMINAL_4B_ITEMS_PER_THREAD;

    using ReducePolicy =
      cub::AgentReducePolicy<nominal_4b_large_threads_per_block,
                             nominal_4b_large_items_per_thread,
                             AccumT,
                             items_per_vec_load,
                             cub::BLOCK_REDUCE_WARP_REDUCTIONS,
                             cub::LOAD_LDG>;

    using SmallReducePolicy =
      cub::AgentWarpReducePolicy<ReducePolicy::BLOCK_THREADS,
                                 small_threads_per_warp,
                                 nominal_4b_small_items_per_thread,
                                 AccumT,
                                 items_per_vec_load,
                                 cub::LOAD_LDG>;

    using MediumReducePolicy =
      cub::AgentWarpReducePolicy<ReducePolicy::BLOCK_THREADS,
                                 medium_threads_per_warp,
                                 nominal_4b_medium_items_per_thread,
                                 AccumT,
                                 items_per_vec_load,
                                 cub::LOAD_LDG>;
  };

  using MaxPolicy = policy_t;
};
#endif // !TUNE_BASE

template <typename T, typename OffsetT, typename MaxSegmentSizeGuaranteeT>
void variable_segmented_reduce(nvbench::state& state, size_t max_seg_size_guarantee)
{
  static constexpr bool is_argmin = std::is_same_v<op_t, cub::detail::arg_min>;
  static constexpr bool is_argmax = std::is_same_v<op_t, cub::detail::arg_max>;

  using raw_input_it_t = const T*;
  using output_t       = cuda::std::conditional_t<(is_argmin || is_argmax), cuda::std::pair<int, T>, T>;
  using output_it_t    = output_t*;
  using accum_t        = output_t;
  using init_t =
    cuda::std::conditional_t<(is_argmin || is_argmax), cub::detail::reduce::empty_problem_init_t<accum_t>, T>;
  using offset_t                     = OffsetT;
  using begin_offset_it_t            = const offset_t*;
  using end_offset_it_t              = const offset_t*;
  using max_segment_size_guarantee_t = MaxSegmentSizeGuaranteeT;

  // Retrieve axis parameters
  const auto elements         = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  const auto max_segment_size = static_cast<std::size_t>(state.get_int64("MaxSegmentSize"));

  // Compute min segment size as half of max (similar to segmented_sort)
  const auto max_segment_size_log = static_cast<offset_t>(std::log2(max_segment_size));
  const auto min_segment_size     = std::max<std::size_t>(1, 1 << (max_segment_size_log - 1));

  // Generate segment offsets
  thrust::device_vector<offset_t> segment_offsets =
    generate.uniform.segment_offsets(elements, min_segment_size, max_segment_size);
  const auto num_segments = segment_offsets.size() - 1;

  // Generate input data
  thrust::device_vector<T> in = generate(elements);
  thrust::device_vector<output_t> out(num_segments);

  raw_input_it_t d_raw_in           = thrust::raw_pointer_cast(in.data());
  output_it_t d_out                 = thrust::raw_pointer_cast(out.data());
  begin_offset_it_t d_begin_offsets = thrust::raw_pointer_cast(segment_offsets.data());
  end_offset_it_t d_end_offsets     = d_begin_offsets + 1;

  // Create wrapped iterator for argmin/argmax operations
  [[maybe_unused]] auto d_indexed_in = thrust::make_transform_iterator(
    thrust::counting_iterator<::cuda::std::int64_t>{0},
    cub::detail::reduce::generate_idx_value<raw_input_it_t, T>(d_raw_in, 1));
  using arg_index_input_iterator_t = decltype(d_indexed_in);

  auto get_in = [&] {
    if constexpr (is_argmin || is_argmax)
    {
      return d_indexed_in;
    }
    else
    {
      return d_raw_in;
    }
  };

  using input_it_t = decltype(get_in());
  input_it_t d_in  = get_in();

  // Enable throughput calculations
  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<T>(num_segments);
  state.add_global_memory_reads<offset_t>(num_segments + 1);

  using dispatch_t = cub::DispatchSegmentedReduce<
    input_it_t,
    output_it_t,
    begin_offset_it_t,
    end_offset_it_t,
    offset_t,
    op_t,
    init_t,
    accum_t,
    max_segment_size_guarantee_t
#if !TUNE_BASE
    ,
    policy_hub_t<accum_t>
#endif // TUNE_BASE
    >;

  auto guarantee_max_segment_size = max_segment_size_guarantee_t{max_seg_size_guarantee};

  // Allocate temporary storage
  std::size_t temp_size{};
  dispatch_t::Dispatch(
    nullptr,
    temp_size,
    d_in,
    d_out,
    static_cast<::cuda::std::int64_t>(num_segments),
    d_begin_offsets,
    d_end_offsets,
    op_t{},
    init_t{},
    0 /* stream */,
    guarantee_max_segment_size);

  thrust::device_vector<nvbench::uint8_t> temp(temp_size);
  auto* temp_storage = thrust::raw_pointer_cast(temp.data());

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    dispatch_t::Dispatch(
      temp_storage,
      temp_size,
      d_in,
      d_out,
      static_cast<::cuda::std::int64_t>(num_segments),
      d_begin_offsets,
      d_end_offsets,
      op_t{},
      init_t{},
      launch.get_stream(),
      guarantee_max_segment_size);
  });
}

template <typename T, typename OffsetT>
void variable_segmented_reduce_default(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  variable_segmented_reduce<T, OffsetT, cuda::execution::max_segment_size<0>>(state, 0);
}

template <typename T, typename OffsetT>
void variable_segmented_reduce_small(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  variable_segmented_reduce<T, OffsetT, cuda::execution::max_segment_size<16>>(state, 16);
}

template <typename T, typename OffsetT>
void variable_segmented_reduce_medium(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  variable_segmented_reduce<T, OffsetT, cuda::execution::max_segment_size<256>>(state, 256);
}

template <typename T, typename OffsetT>
void variable_segmented_reduce_large(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  variable_segmented_reduce<T, OffsetT, cuda::execution::max_segment_size<16384>>(state, 16384);
}

template <typename T, typename OffsetT>
void variable_segmented_reduce_dynamic_small(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  variable_segmented_reduce<T, OffsetT, cuda::execution::max_segment_size<>>(state, 16);
}

template <typename T, typename OffsetT>
void variable_segmented_reduce_dynamic_medium(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  variable_segmented_reduce<T, OffsetT, cuda::execution::max_segment_size<>>(state, 256);
}

template <typename T, typename OffsetT>
void variable_segmented_reduce_dynamic_large(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  variable_segmented_reduce<T, OffsetT, cuda::execution::max_segment_size<>>(state, 16384);
}

/* Default max segment size=0 benchmarks, performing block reduction per segment */

NVBENCH_BENCH_TYPES(variable_segmented_reduce_default, NVBENCH_TYPE_AXES(value_types, some_offset_types))
  .set_name("variable_default")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_int64_power_of_two_axis("MaxSegmentSize", nvbench::range(1, 16, 1));

/* STATIC max segment size benchmarks */

// Small segments: 1-16 items per segment
NVBENCH_BENCH_TYPES(variable_segmented_reduce_small, NVBENCH_TYPE_AXES(value_types, some_offset_types))
  .set_name("variable_small")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_int64_power_of_two_axis("MaxSegmentSize", nvbench::range(1, 4, 1));

// Medium segments: 32-256 items per segment
NVBENCH_BENCH_TYPES(variable_segmented_reduce_medium, NVBENCH_TYPE_AXES(value_types, some_offset_types))
  .set_name("variable_medium")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_int64_power_of_two_axis("MaxSegmentSize", nvbench::range(5, 8, 1));

// Large segments: 512+ items per segment
NVBENCH_BENCH_TYPES(variable_segmented_reduce_large, NVBENCH_TYPE_AXES(value_types, some_offset_types))
  .set_name("variable_large")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_int64_power_of_two_axis("MaxSegmentSize", nvbench::range(9, 16, 1));

/* DYNAMIC max segment size benchmarks */

// Small segments: 1-16 items per segment
NVBENCH_BENCH_TYPES(variable_segmented_reduce_dynamic_small, NVBENCH_TYPE_AXES(value_types, some_offset_types))
  .set_name("variable_small_dynamic")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_int64_power_of_two_axis("MaxSegmentSize", nvbench::range(1, 4, 1));

// Medium segments: 32-256 items per segment
NVBENCH_BENCH_TYPES(variable_segmented_reduce_dynamic_medium, NVBENCH_TYPE_AXES(value_types, some_offset_types))
  .set_name("variable_medium_dynamic")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_int64_power_of_two_axis("MaxSegmentSize", nvbench::range(5, 8, 1));

// Large segments: 512+ items per segment
NVBENCH_BENCH_TYPES(variable_segmented_reduce_dynamic_large, NVBENCH_TYPE_AXES(value_types, some_offset_types))
  .set_name("variable_large_dynamic")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_int64_power_of_two_axis("MaxSegmentSize", nvbench::range(9, 16, 1));
