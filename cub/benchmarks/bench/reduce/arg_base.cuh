// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cub/device/device_reduce.cuh>
#include <cub/device/dispatch/dispatch_streaming_reduce.cuh>

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
  };

  using MaxPolicy = policy_t;
};
#endif // !TUNE_BASE

template <typename T, typename OffsetT>
void arg_reduce(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  using offset_t = OffsetT;

  // Offset type used within the kernel and to index within one partition
  using per_partition_offset_t = int;

  // Offset type used to index within the total input in the range [d_in, d_in + num_items)
  using global_offset_t = ::cuda::std::uint64_t;

  // The value type of the KeyValuePair<offset_t, output_value_t> returned by the ArgIndexInputIterator
  using output_value_t = T;

  // Iterator providing the values being reduced
  using values_it_t = T*;

  // Iterator providing the input items for the reduction
  using input_it_t = values_it_t;

  // Type used for the final result
  using output_tuple_t = cub::KeyValuePair<global_offset_t, T>;

  auto const init = ::cuda::std::numeric_limits<T>::max();

#if !TUNE_BASE
  using policy_t   = policy_hub_t<accum_t, offset_t>;
  using dispatch_t = cub::DispatchReduce<input_it_t, output_it_t, offset_t, op_t, init_t, accum_t, policy_t>;
#else // TUNE_BASE
  using dispatch_t =
    cub::DispatchStreamingArgReduce<input_it_t, T*, global_offset_t*, per_partition_offset_t, global_offset_t, op_t, T>;
#endif // TUNE_BASE

  // Retrieve axis parameters
  const auto elements         = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  thrust::device_vector<T> in = generate(elements);
  thrust::device_vector<output_tuple_t> out(1);

  values_it_t d_in             = thrust::raw_pointer_cast(in.data());
  global_offset_t* d_index_out = &(thrust::raw_pointer_cast(out.data()))->key;
  output_value_t* d_result_out = &(thrust::raw_pointer_cast(out.data()))->value;

  // Enable throughput calculations and add "Size" column to results.
  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<output_tuple_t>(1);

  // Allocate temporary storage:
  std::size_t temp_size;
  dispatch_t::Dispatch(
    nullptr, temp_size, d_in, d_result_out, d_index_out, static_cast<offset_t>(elements), op_t{}, init, 0 /* stream */);

  thrust::device_vector<nvbench::uint8_t> temp(temp_size);
  auto* temp_storage = thrust::raw_pointer_cast(temp.data());

  state.exec(nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    dispatch_t::Dispatch(
      temp_storage,
      temp_size,
      d_in,
      d_result_out,
      d_index_out,
      static_cast<offset_t>(elements),
      op_t{},
      init,
      launch.get_stream());
  });
}

NVBENCH_BENCH_TYPES(arg_reduce, NVBENCH_TYPE_AXES(fundamental_types, offset_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
