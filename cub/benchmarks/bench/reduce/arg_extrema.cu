// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cub/device/device_reduce.cuh>
#include <cub/device/dispatch/tuning/tuning_reduce.cuh>

#include <cuda/std/type_traits>

#include <nvbench_helper.cuh>

// %RANGE% TUNE_ITEMS_PER_THREAD ipt 7:24:1
// %RANGE% TUNE_THREADS_PER_BLOCK tpb 128:1024:32
// %RANGE% TUNE_ITEMS_PER_VEC_LOAD_POW2 ipv 1:2:1

#if !TUNE_BASE
struct tuned_policy_selector
{
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr auto operator()(cuda::compute_capability) const
    -> cub::detail::reduce::reduce_policy
  {
    cub::detail::reduce::agent_reduce_policy rp{
      TUNE_THREADS_PER_BLOCK,
      TUNE_ITEMS_PER_THREAD,
      1 << TUNE_ITEMS_PER_VEC_LOAD_POW2,
      cub::BLOCK_REDUCE_WARP_REDUCTIONS,
      cub::LOAD_DEFAULT};
    auto rp_nondet            = rp;
    rp_nondet.block_algorithm = cub::BLOCK_REDUCE_WARP_REDUCTIONS_NONDETERMINISTIC;
    return {rp, rp};
  }
};
#endif // !TUNE_BASE

template <typename T, typename OpT>
void arg_reduce(nvbench::state& state, nvbench::type_list<T, OpT>)
{
  // Offset type used to index within the total input in the range [d_in, d_in + num_items)
  using offset_t = cuda::std::int64_t;

  // Retrieve axis parameters
  const auto elements         = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  thrust::device_vector<T> in = generate(elements);
  thrust::device_vector<offset_t> out_index(1);
  thrust::device_vector<T> out_extremum(1);

  const T* d_in         = thrust::raw_pointer_cast(in.data());
  offset_t* d_out_index = thrust::raw_pointer_cast(out_index.data());
  T* d_out_extremum     = thrust::raw_pointer_cast(out_extremum.data());

  // Enable throughput calculations and add "Size" column to results.
  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<offset_t>(1);
  state.add_global_memory_writes<T>(1);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    auto env = cub_bench_env(
      alloc,
      launch
#if !TUNE_BASE
      ,
      cuda::execution::tune(tuned_policy_selector{})
#endif // !TUNE_BASE
    );
    if constexpr (cuda::std::is_same_v<OpT, cub::detail::arg_min>)
    {
      _CCCL_TRY_CUDA_API(
        cub::DeviceReduce::ArgMin,
        "ArgMin failed",
        d_in,
        d_out_extremum,
        d_out_index,
        static_cast<offset_t>(elements),
        cuda::std::less{},
        env);
    }
    else
    {
      _CCCL_TRY_CUDA_API(
        cub::DeviceReduce::ArgMax,
        "ArgMax failed",
        d_in,
        d_out_extremum,
        d_out_index,
        static_cast<offset_t>(elements),
        cuda::std::less{},
        env);
    }
  });
}

using op_types = nvbench::type_list<cub::detail::arg_min, cub::detail::arg_max>;

NVBENCH_BENCH_TYPES(arg_reduce, NVBENCH_TYPE_AXES(fundamental_types, op_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}", "Operation{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
