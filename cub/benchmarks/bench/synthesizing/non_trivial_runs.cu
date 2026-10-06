// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#define CCCL_IGNORE_DEPRECATED_API

#include <cub/device/device_run_length_encode.cuh>

#include <nvbench_helper.cuh>

#include "iterators.cuh"

// Non-trivial-run encode's tuned policy requests a warp-transpose BlockLoad. The input is synthesizing, so that load
// does not read memory. A counting sequence changes on every item, so the scan finds no run longer than one item.
// The offset and length buffers are sized for the worst case, one run per item.
template <class Input>
void non_trivial_runs(nvbench::state& state, nvbench::type_list<Input>)
try
{
  const auto elements = static_cast<::cuda::std::int64_t>(state.get_int64("Elements{io}"));
  auto input          = Input::make(static_cast<::cuda::std::int64_t>(elements));
  auto offsets        = synthesizing_make_output<::cuda::std::int64_t>(static_cast<std::size_t>(elements));
  auto lengths        = synthesizing_make_output<::cuda::std::int64_t>(static_cast<std::size_t>(elements));
  thrust::device_vector<::cuda::std::int64_t> num_runs(1, thrust::no_init);

  state.add_element_count(elements);
  state.add_global_memory_writes<::cuda::std::int64_t>(1);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    auto env = cub_bench_env(alloc, launch);
    _CCCL_TRY_RUNTIME_API(
      cub::DeviceRunLengthEncode::NonTrivialRuns,
      "NonTrivialRuns failed",
      input,
      offsets.begin(),
      lengths.begin(),
      num_runs.begin(),
      elements,
      env);
  });
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

NVBENCH_BENCH_TYPES(non_trivial_runs, NVBENCH_TYPE_AXES(synthesizing_input_types))
  .set_name("base")
  .set_type_axes_names({"Input{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
