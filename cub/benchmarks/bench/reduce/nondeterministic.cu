// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cub/device/dispatch/dispatch_reduce_nondeterministic.cuh>

#include <nvbench_helper.cuh>

#include <nvbench/range.cuh>
#include <nvbench/types.cuh>

// %RANGE% TUNE_ITEMS_PER_THREAD ipt 3:24:1
// %RANGE% TUNE_THREADS_PER_BLOCK tpb 128:1024:32
// %RANGE% TUNE_ITEMS_PER_VEC_LOAD_POW2 ipv 1:2:1

#ifndef TUNE_BASE
#  define TUNE_ITEMS_PER_VEC_LOAD (1 << TUNE_ITEMS_PER_VEC_LOAD_POW2)
#endif

#if !TUNE_BASE
template <typename AccumT, typename OffsetT>
struct policy_hub_t
{
  struct policy_t : cub::ChainedPolicy<300, policy_t, policy_t>
  {
    static constexpr int threads_per_block  = TUNE_THREADS_PER_BLOCK;
    static constexpr int items_per_thread   = TUNE_ITEMS_PER_THREAD;
    static constexpr int items_per_vec_load = TUNE_ITEMS_PER_VEC_LOAD;

    using ReducePolicy =
      cub::AgentReducePolicy<threads_per_block,
                             items_per_thread,
                             AccumT,
                             items_per_vec_load,
                             cub::BLOCK_REDUCE_WARP_REDUCTIONS,
                             cub::LOAD_DEFAULT>;

    // SingleTilePolicy
    using SingleTilePolicy = ReducePolicy;

    // SegmentedReducePolicy
    using SegmentedReducePolicy = ReducePolicy;

    // ReduceNondeterministicPolicy
    using ReduceNondeterministicPolicy = ReducePolicy;
  };

  using MaxPolicy = policy_t;
};
#endif // !TUNE_BASE

template <typename T, typename OffsetT>
void nondeterministic_sum(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  using accum_t     = T;
  using input_it_t  = const T*;
  using output_it_t = T*;
  using offset_t    = cub::detail::choose_offset_t<OffsetT>;
  using output_t    = T;
  using op_t        = cuda::std::plus<>;
  using init_t      = T;
  using dispatch_t  = cub::detail::DispatchReduceNondeterministic<
     input_it_t,
     output_it_t,
     offset_t,
     op_t,
     init_t,
     accum_t
#if !TUNE_BASE
    ,
    policy_hub_t<accum_t, offset_t>
#endif // TUNE_BASE
    >;

  // Retrieve axis parameters
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements{io}"));

  thrust::device_vector<T> in = generate(elements);
  thrust::device_vector<T> out(1);

  input_it_t d_in   = thrust::raw_pointer_cast(in.data());
  output_it_t d_out = thrust::raw_pointer_cast(out.data());

  // Enable throughput calculations and add "Size" column to results.
  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<T>(1);

  // Allocate temporary storage:
  std::size_t temp_size;
  dispatch_t::Dispatch(
    nullptr, temp_size, d_in, d_out, static_cast<offset_t>(elements), op_t{}, init_t{}, 0 /* stream */);

  thrust::device_vector<nvbench::uint8_t> temp(temp_size, thrust::no_init);
  auto* temp_storage = thrust::raw_pointer_cast(temp.data());

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    dispatch_t::Dispatch(
      temp_storage, temp_size, d_in, d_out, static_cast<offset_t>(elements), op_t{}, init_t{}, launch.get_stream());
  });
}

#ifdef TUNE_T
using value_types = nvbench::type_list<TUNE_T>;
#else
using value_types = nvbench::type_list<int32_t, int64_t, float, double>;
#endif

NVBENCH_BENCH_TYPES(nondeterministic_sum, NVBENCH_TYPE_AXES(value_types, offset_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
