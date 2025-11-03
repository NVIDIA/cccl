// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/device/device_reduce.cuh>

#include <cuda/std/type_traits>

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

template <typename T>
void fixed_size_segmented_reduce(nvbench::state& state, nvbench::type_list<T>)
{
  static constexpr bool is_argmin = std::is_same_v<op_t, cub::detail::arg_min>;

  using input_it_t  = const T*;
  using output_t    = cuda::std::conditional_t<is_argmin, cuda::std::pair<int, T>, T>;
  using output_it_t = output_t*;
  using accum_t     = output_t;
  using init_t      = cuda::std::conditional_t<is_argmin, cub::detail::reduce::empty_problem_init_t<accum_t>, T>;

  // Retrieve axis parameters
  const size_t num_elements = static_cast<size_t>(state.get_int64("Elements{io}"));
  const size_t segment_size = static_cast<size_t>(state.get_int64("SegmentSize"));
  const size_t num_segments = std::max<std::size_t>(1, (num_elements / segment_size));
  const size_t elements     = num_segments * segment_size;

  thrust::device_vector<T> in = generate(elements);
  thrust::device_vector<output_t> out(num_segments);

  input_it_t d_in   = thrust::raw_pointer_cast(in.data());
  output_it_t d_out = thrust::raw_pointer_cast(out.data());

  // Enable throughput calculations and add "Size" column to results.
  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<output_t>(num_segments);

  [[maybe_unused]] auto d_indexed_in = thrust::make_transform_iterator(
    thrust::counting_iterator<::cuda::std::int64_t>{0},
    cub::detail::reduce::generate_idx_value<input_it_t, T>(d_in, segment_size));
  using arg_index_input_iterator_t = decltype(d_indexed_in);

  auto select_in = [&] {
    if constexpr (is_argmin)
    {
      return d_indexed_in;
    }
    else
    {
      return d_in;
    }
  };

  using dispatch_t = cub::detail::reduce::DispatchFixedSizeSegmentedReduce<
    cuda::std::conditional_t<is_argmin, arg_index_input_iterator_t, input_it_t>,
    output_it_t,
    int,
    op_t,
    init_t,
    accum_t
#if !TUNE_BASE
    ,
    policy_hub_t<accum_t>
#endif // TUNE_BASE
    >;

  // Allocate temporary storage:
  std::size_t temp_size;
  dispatch_t::Dispatch(
    nullptr,
    temp_size,
    select_in(),
    d_out,
    num_segments,
    static_cast<int>(segment_size),
    op_t{},
    init_t{},
    0 /* stream */);

  thrust::device_vector<nvbench::uint8_t> temp(temp_size);
  auto* temp_storage = thrust::raw_pointer_cast(temp.data());

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    dispatch_t::Dispatch(
      temp_storage,
      temp_size,
      select_in(),
      d_out,
      num_segments,
      static_cast<int>(segment_size),
      op_t{},
      init_t{},
      launch.get_stream());
  });
}

NVBENCH_BENCH_TYPES(fixed_size_segmented_reduce, NVBENCH_TYPE_AXES(value_types))
  .set_name("small")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_int64_power_of_two_axis("SegmentSize", nvbench::range(0, 4, 1));

NVBENCH_BENCH_TYPES(fixed_size_segmented_reduce, NVBENCH_TYPE_AXES(value_types))
  .set_name("medium")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_int64_power_of_two_axis("SegmentSize", nvbench::range(5, 8, 1));

NVBENCH_BENCH_TYPES(fixed_size_segmented_reduce, NVBENCH_TYPE_AXES(value_types))
  .set_name("large")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_int64_power_of_two_axis("SegmentSize", nvbench::range(9, 16, 1));
