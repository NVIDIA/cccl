// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define CCCL_IGNORE_DEPRECATED_API

#include <cub/device/device_reduce.cuh>

#include <nvbench_helper.cuh>

#include "iterators.cuh"

// Reduce-by-key loads keys and values through one BlockLoad algorithm. On current architectures that algorithm is a
// warp-transpose load for most integer key and value widths. Both iterators here are synthesizing, so neither load
// reads memory. Keys are the item index divided by a fixed run length.
inline constexpr ::cuda::std::int64_t synthesizing_reduce_by_key_run_length = 4096;

struct synthesizing_reduce_by_key_key_t
{
  _CCCL_HOST_DEVICE constexpr ::cuda::std::int64_t operator()(::cuda::std::int64_t index) const noexcept
  {
    return index / synthesizing_reduce_by_key_run_length;
  }
};

template <class Input>
void reduce_by_key(nvbench::state& state, nvbench::type_list<Input>)
try
{
  const auto elements = static_cast<::cuda::std::int64_t>(state.get_int64("Elements{io}"));
  const auto num_runs = (elements + synthesizing_reduce_by_key_run_length - 1) / synthesizing_reduce_by_key_run_length;
  auto values         = Input::make(elements);
  using value_t       = cub::detail::it_value_t<decltype(values)>;
  auto unique_keys    = synthesizing_make_output<::cuda::std::int64_t>(static_cast<std::size_t>(num_runs));
  auto aggregates     = synthesizing_make_output<value_t>(static_cast<std::size_t>(num_runs));
  thrust::device_vector<::cuda::std::int64_t> num_runs_out(1, thrust::no_init);
  const auto keys =
    cuda::transform_iterator{cuda::counting_iterator<::cuda::std::int64_t>{0}, synthesizing_reduce_by_key_key_t{}};

  state.add_element_count(elements);
  state.add_global_memory_writes<::cuda::std::int64_t>(num_runs + 1);
  state.add_global_memory_writes<value_t>(num_runs);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    auto env = cub_bench_env(alloc, launch);
    if constexpr (Input::scalar)
    {
      _CCCL_TRY_RUNTIME_API(
        cub::DeviceReduce::ReduceByKey,
        "ReduceByKey failed",
        keys,
        unique_keys.begin(),
        values,
        aggregates.begin(),
        num_runs_out.begin(),
        ::cuda::std::plus<>{},
        elements,
        env);
    }
    else
    {
      _CCCL_TRY_RUNTIME_API(
        cub::DeviceReduce::ReduceByKey,
        "ReduceByKey failed",
        keys,
        unique_keys.begin(),
        values,
        aggregates.begin(),
        num_runs_out.begin(),
        synthesizing_tuple_plus_t<typename Input::value_type>{},
        elements,
        env);
    }
  });
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

NVBENCH_BENCH_TYPES(reduce_by_key, NVBENCH_TYPE_AXES(synthesizing_input_types))
  .set_name("base")
  .set_type_axes_names({"Input{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
