// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <cub/device/device_reduce.cuh>

#include <nvbench_helper.cuh>

// %RANGE% TUNE_ITEMS_PER_THREAD ipt 7:24:1
// %RANGE% TUNE_THREADS_PER_BLOCK tpb 128:1024:32
// %RANGE% TUNE_ITEMS_PER_VEC_LOAD_POW2 ipv 1:2:1

#if !TUNE_BASE
template <typename AccumT>
struct policy_selector
{
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr auto operator()(cuda::compute_capability) const
    -> cub::detail::reduce::reduce_policy
  {
    const auto [items, threads] =
      cub::detail::scale_mem_bound(TUNE_THREADS_PER_BLOCK, TUNE_ITEMS_PER_THREAD, int{sizeof(AccumT)});
    const auto policy = cub::detail::reduce::agent_reduce_policy{
      threads, items, 1 << TUNE_ITEMS_PER_VEC_LOAD_POW2, cub::BLOCK_REDUCE_WARP_REDUCTIONS, cub::LOAD_DEFAULT};
    return {policy, policy};
  }
};
#endif // !TUNE_BASE

template <class T>
struct square_t
{
  __host__ __device__ T operator()(const T& x) const
  {
    return x * x;
  }
};

template <typename T, typename OffsetT>
void reduce(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  using init_value_t   = T;
  using reduction_op_t = ::cuda::std::plus<>;
  using transform_op_t = square_t<T>;

  // Retrieve axis parameters
  const auto elements = state.get_int64("Elements{io}");

  thrust::device_vector<T> in = generate(elements);
  thrust::device_vector<T> out(1, thrust::default_init);

  auto d_in  = thrust::raw_pointer_cast(in.data());
  auto d_out = thrust::raw_pointer_cast(out.data());

  // Enable throughput calculations and add "Size" column to results.
  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<T>(1);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    auto env = cub_bench_env(
      alloc,
      launch
#if !TUNE_BASE
      ,
      cuda::execution::tune(policy_selector<cuda::std::__accumulator_t<reduction_op_t, T, init_value_t>>{})
#endif // !TUNE_BASE
    );
    _CCCL_TRY_CUDA_API(
      cub::DeviceReduce::TransformReduce,
      "TransformReduce failed",
      d_in,
      d_out,
      static_cast<OffsetT>(elements),
      reduction_op_t{},
      transform_op_t{},
      init_value_t{},
      env);
  });
}

NVBENCH_BENCH_TYPES(reduce, NVBENCH_TYPE_AXES(all_types, offset_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
