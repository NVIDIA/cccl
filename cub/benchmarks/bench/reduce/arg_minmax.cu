// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_reduce.cuh>
#include <cub/device/dispatch/tuning/tuning_reduce.cuh>

#include <nvbench_helper.cuh>

// %RANGE% TUNE_ITEMS_PER_THREAD ipt 7:24:1
// %RANGE% TUNE_THREADS_PER_BLOCK tpb 128:1024:32
// %RANGE% TUNE_ITEMS_PER_VEC_LOAD_POW2 ipv 1:2:1

#if !TUNE_BASE
struct tuned_policy_selector
{
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr auto operator()(cuda::compute_capability) const -> cub::ReducePolicy
  {
    cub::ReducePassPolicy rp{
      TUNE_THREADS_PER_BLOCK,
      TUNE_ITEMS_PER_THREAD,
      1 << TUNE_ITEMS_PER_VEC_LOAD_POW2,
      cub::BLOCK_REDUCE_WARP_REDUCTIONS,
      cub::LOAD_DEFAULT};
    return {rp, rp};
  }
};
#endif // !TUNE_BASE

template <typename T>
void arg_minmax(nvbench::state& state, nvbench::type_list<T>)
{
  using offset_t = cuda::std::int64_t;

  const auto elements         = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  thrust::device_vector<T> in = generate(elements);
  thrust::device_vector<T> out_extrema(2); // [0] = min, [1] = max
  thrust::device_vector<offset_t> out_indices(2); // [0] = min index, [1] = max index

  const T* d_in           = thrust::raw_pointer_cast(in.data());
  offset_t* d_out_indices = thrust::raw_pointer_cast(out_indices.data());
  T* d_out_extrema        = thrust::raw_pointer_cast(out_extrema.data());

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<T>(2);
  state.add_global_memory_writes<offset_t>(2);

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
    _CCCL_TRY_CUDA_API(
      cub::DeviceReduce::ArgMinLastMax,
      "ArgMinLastMax failed",
      d_in,
      d_out_extrema,
      d_out_indices,
      d_out_extrema + 1,
      d_out_indices + 1,
      static_cast<offset_t>(elements),
      env);
  });
}

NVBENCH_BENCH_TYPES(arg_minmax, NVBENCH_TYPE_AXES(fundamental_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
