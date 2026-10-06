// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#define CCCL_IGNORE_DEPRECATED_API

#include <cub/device/device_select.cuh>

#include <nvbench_helper.cuh>

#include "iterators.cuh"

// Unique-by-key loads keys and values through one BlockLoad algorithm. The default policy, and several tuned
// policies, request a warp-transpose load. Both iterators here are synthesizing, so neither load reads memory. Keys
// are the item index divided by a fixed run length, so each run keeps one key and one value.
inline constexpr ::cuda::std::int64_t synthesizing_unique_by_key_run_length = 4096;

struct synthesizing_unique_by_key_key_t
{
  _CCCL_HOST_DEVICE constexpr ::cuda::std::int64_t operator()(::cuda::std::int64_t index) const noexcept
  {
    return index / synthesizing_unique_by_key_run_length;
  }
};

template <class Input>
void unique_by_key(nvbench::state& state, nvbench::type_list<Input>)
try
{
  const auto elements = static_cast<::cuda::std::int64_t>(state.get_int64("Elements{io}"));
  const auto num_runs = (elements + synthesizing_unique_by_key_run_length - 1) / synthesizing_unique_by_key_run_length;
  auto values         = Input::make(static_cast<::cuda::std::int64_t>(elements));
  using value_t       = cub::detail::it_value_t<decltype(values)>;
  auto unique_keys    = synthesizing_make_output<::cuda::std::int64_t>(static_cast<std::size_t>(num_runs));
  auto unique_values  = synthesizing_make_output<value_t>(static_cast<std::size_t>(num_runs));
  thrust::device_vector<::cuda::std::int64_t> num_selected(1, thrust::no_init);
  const auto keys = cuda::transform_iterator{
    cuda::counting_iterator<::cuda::std::int64_t>{0}, synthesizing_unique_by_key_key_t{}};

  state.add_element_count(elements);
  state.add_global_memory_writes<::cuda::std::int64_t>(num_runs + 1);
  state.add_global_memory_writes<value_t>(num_runs);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    auto env = cub_bench_env(alloc, launch);
    _CCCL_TRY_RUNTIME_API(
      cub::DeviceSelect::UniqueByKey,
      "UniqueByKey failed",
      keys,
      values,
      unique_keys.begin(),
      unique_values.begin(),
      num_selected.begin(),
      elements,
      ::cuda::std::equal_to<>{},
      env);
  });
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

NVBENCH_BENCH_TYPES(unique_by_key, NVBENCH_TYPE_AXES(synthesizing_input_types))
  .set_name("base")
  .set_type_axes_names({"Input{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
