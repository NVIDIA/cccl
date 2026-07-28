// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_reduce.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>

#include <cuda/argument>
#include <cuda/execution.determinism.h>
#include <cuda/execution.require.h>
#include <cuda/std/functional>

#include <cstddef>

#include <nvbench_helper.cuh>

#include <nvbench/range.cuh>
#include <nvbench/types.cuh>

// %RANGE% TUNE_ITEMS_PER_THREAD ipt 3:24:1
// %RANGE% TUNE_THREADS_PER_BLOCK tpb 128:1024:32
// %RANGE% TUNE_ITEMS_PER_VEC_LOAD_POW2 ipv 1:2:1

#if !TUNE_BASE
template <typename AccumT>
struct policy_selector
{
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr auto operator()(cuda::compute_capability) const -> cub::ReducePolicy
  {
    const auto [items, threads] =
      cub::detail::scale_mem_bound(TUNE_THREADS_PER_BLOCK, TUNE_ITEMS_PER_THREAD, int{sizeof(AccumT)});
    const auto policy = cub::ReducePassPolicy{
      threads,
      items,
      1 << TUNE_ITEMS_PER_VEC_LOAD_POW2,
      cub::BLOCK_REDUCE_WARP_REDUCTIONS_NONDETERMINISTIC,
      cub::LOAD_DEFAULT};
    return {policy, {}};
  }
};
#endif // !TUNE_BASE

template <typename T, typename OffsetT>
void nondeterministic_sum(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  using op_t         = cuda::std::plus<>;
  using init_value_t = T;

  // Retrieve axis parameters
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements{io}"));

  thrust::device_vector<T> in = generate(elements);
  thrust::device_vector<T> out(1, thrust::no_init);
  thrust::device_vector<OffsetT> device_num_items(1, static_cast<OffsetT>(elements));

  auto d_in        = thrust::raw_pointer_cast(in.data());
  auto d_out       = thrust::raw_pointer_cast(out.data());
  auto d_num_items = thrust::raw_pointer_cast(device_num_items.data());

  // Enable throughput calculations and add "Size" column to results.
  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<T>(1);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    auto env = cub_bench_env(
      alloc,
      launch,
      cuda::execution::require(cuda::execution::determinism::not_guaranteed)
#if !TUNE_BASE
        ,
      cuda::execution::tune(policy_selector<cuda::std::__accumulator_t<op_t, T, init_value_t>>{})
#endif // !TUNE_BASE
    );
    _CCCL_TRY_CUDA_API(
      cub::DeviceReduce::Reduce,
      "Reduce failed",
      d_in,
      d_out,
      cuda::args::deferred{d_num_items},
      op_t{},
      init_value_t{},
      env);
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
