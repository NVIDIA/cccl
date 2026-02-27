// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cub/device/device_reduce.cuh>
#include <cub/device/dispatch/dispatch_streaming_reduce.cuh>
#include <cub/device/dispatch/tuning/tuning_reduce.cuh>

#include <cuda/__device/arch_id.h>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include <nvbench_helper.cuh>

// %RANGE% TUNE_ITEMS_PER_THREAD ipt 7:24:1
// %RANGE% TUNE_THREADS_PER_BLOCK tpb 128:1024:32
// %RANGE% TUNE_ITEMS_PER_VEC_LOAD_POW2 ipv 1:2:1

#if !TUNE_BASE
struct tuned_policy_selector
{
  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::arch_id) const -> cub::detail::reduce::reduce_policy
  {
    cub::detail::reduce::agent_reduce_policy rp{
      TUNE_THREADS_PER_BLOCK,
      TUNE_ITEMS_PER_THREAD,
      1 << TUNE_ITEMS_PER_VEC_LOAD_POW2,
      cub::BLOCK_REDUCE_WARP_REDUCTIONS,
      cub::LOAD_DEFAULT};
    auto rp_nondet            = rp;
    rp_nondet.block_algorithm = cub::BLOCK_REDUCE_WARP_REDUCTIONS_NONDETERMINISTIC;
    return {rp, rp, rp_nondet};
  }
};
#endif // !TUNE_BASE

template <typename T, typename OpT>
void arg_reduce(nvbench::state& state, nvbench::type_list<T, OpT>)
{
  // Offset type used within the kernel and to index within one partition
  using per_partition_offset_t = int;

  // Offset type used to index within the total input in the range [d_in, d_in + num_items)
  using global_offset_t = ::cuda::std::int64_t;

  // Iterator providing the values being reduced
  using values_it_t = T*;

  // Type used for the final result
  using output_tuple_t = cub::KeyValuePair<global_offset_t, T>;

  auto const init = ::cuda::std::is_same_v<OpT, cub::ArgMin>
                    ? ::cuda::std::numeric_limits<T>::max()
                    : ::cuda::std::numeric_limits<T>::lowest();

  // Retrieve axis parameters
  const auto elements         = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  thrust::device_vector<T> in = generate(elements);
  thrust::device_vector<output_tuple_t> out(1);

  values_it_t d_in      = thrust::raw_pointer_cast(in.data());
  output_tuple_t* d_out = thrust::raw_pointer_cast(out.data());
  auto const num_items  = static_cast<global_offset_t>(elements);

  // Enable throughput calculations and add "Size" column to results.
  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<output_tuple_t>(1);

  // Allocate temporary storage
  std::size_t temp_size;
  cub::detail::reduce::dispatch_streaming_arg_reduce<per_partition_offset_t>(
    nullptr,
    temp_size,
    d_in,
    d_out,
    num_items,
    OpT{},
    init,
    0 /* stream */
#if !TUNE_BASE
    ,
    tuned_policy_selector{}
#endif // TUNE_BASE
  );

  thrust::device_vector<nvbench::uint8_t> temp(temp_size);
  auto* temp_storage = thrust::raw_pointer_cast(temp.data());

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    cub::detail::reduce::dispatch_streaming_arg_reduce<per_partition_offset_t>(
      temp_storage,
      temp_size,
      d_in,
      d_out,
      num_items,
      OpT{},
      init,
      launch.get_stream()
#if !TUNE_BASE
        ,
      tuned_policy_selector{}
#endif // TUNE_BASE
    );
  });
}

using op_types = nvbench::type_list<cub::ArgMin, cub::ArgMax>;

NVBENCH_BENCH_TYPES(arg_reduce, NVBENCH_TYPE_AXES(fundamental_types, op_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}", "Operation{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
