// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <cub/device/device_reduce.cuh>

#include <cuda/execution.determinism.h>
#include <cuda/execution.require.h>

#include <nvbench_helper.cuh>

#include <nvbench/range.cuh>
#include <nvbench/types.cuh>

// %RANGE% TUNE_ITEMS_PER_THREAD ipt 3:24:1
// %RANGE% TUNE_THREADS_PER_BLOCK tpb 128:1024:32

#if !TUNE_BASE
struct policy_selector_t
{
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr auto operator()(cuda::compute_capability) const
    -> cub::detail::rfa::rfa_policy
  {
    return {{TUNE_THREADS_PER_BLOCK, TUNE_ITEMS_PER_THREAD, cub::BLOCK_REDUCE_RAKING},
            {TUNE_THREADS_PER_BLOCK, TUNE_ITEMS_PER_THREAD, cub::BLOCK_REDUCE_RAKING}};
  }
};
#endif // !TUNE_BASE

template <class T>
void deterministic_sum(nvbench::state& state, nvbench::type_list<T>)
{
  using init_t = T;

  const auto elements = static_cast<int>(state.get_int64("Elements{io}"));

  thrust::device_vector<T> in = generate(elements);
  thrust::device_vector<T> out(1);

  const T* d_in = thrust::raw_pointer_cast(in.data());
  T* d_out      = thrust::raw_pointer_cast(out.data());
  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<T>(out.size());

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::no_batch | nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto env = cub_bench_env(
      alloc,
      launch,
      cuda::execution::require(cuda::execution::determinism::gpu_to_gpu)
#if !TUNE_BASE
        ,
      cuda::execution::tune(policy_selector_t{})
#endif // !TUNE_BASE
    );
    _CCCL_TRY_CUDA_API(
      cub::DeviceReduce::Reduce, "Reduce failed", d_in, d_out, elements, cuda::std::plus<>{}, init_t{}, env);
  });
}

using types = nvbench::type_list<float, double>;
NVBENCH_BENCH_TYPES(deterministic_sum, NVBENCH_TYPE_AXES(types))
  .set_name("base")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
