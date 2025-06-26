/******************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include <cub/device/device_reduce.cuh>

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
  using accum_t     = T;
  using input_it_t  = const T*;
  using output_it_t = T*;
  using init_t      = T;

  using dispatch_t = cub::detail::reduce::DispatchFixedSizeSegmentedReduce<
    input_it_t,
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

  // Retrieve axis parameters
  const size_t num_elements = static_cast<size_t>(state.get_int64("Elements{io}"));
  const size_t segment_size = static_cast<size_t>(state.get_int64("SegmentSize"));
  const size_t num_segments = std::max<std::size_t>(1, (num_elements / segment_size));
  const size_t elements     = num_segments * segment_size;

  thrust::device_vector<T> in = generate(elements);
  thrust::device_vector<T> out(num_segments);

  input_it_t d_in   = thrust::raw_pointer_cast(in.data());
  output_it_t d_out = thrust::raw_pointer_cast(out.data());

  // Enable throughput calculations and add "Size" column to results.
  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<T>(num_segments);

  // Allocate temporary storage:
  std::size_t temp_size;
  dispatch_t::Dispatch(
    nullptr, temp_size, d_in, d_out, num_segments, static_cast<int>(segment_size), op_t{}, init_t{}, 0 /* stream */);

  thrust::device_vector<nvbench::uint8_t> temp(temp_size);
  auto* temp_storage = thrust::raw_pointer_cast(temp.data());

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    dispatch_t::Dispatch(
      temp_storage,
      temp_size,
      d_in,
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
