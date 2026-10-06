// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define CCCL_IGNORE_DEPRECATED_API

#include <cub/device/device_select.cuh>

#include <nvbench_helper.cuh>

#include "iterators.cuh"

// Select even values from a synthesizing input. Counting and zip inputs keep about half of the items. A transform that
// adds one also keeps half. A stride of two produces only even values, so that input is kept in full.
template <class Input>
void select_if(nvbench::state& state, nvbench::type_list<Input>)
try
{
  const auto elements     = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  const auto num_selected = Input::num_selected(static_cast<::cuda::std::int64_t>(elements));
  auto input              = Input::make(static_cast<::cuda::std::int64_t>(elements));
  using value_t           = cub::detail::it_value_t<decltype(input)>;
  auto output             = synthesizing_make_output<value_t>(elements);
  thrust::device_vector<::cuda::std::int64_t> selected_count(1, thrust::no_init);

  state.add_element_count(elements);
  state.add_global_memory_writes<value_t>(num_selected);
  state.add_global_memory_writes<::cuda::std::int64_t>(1);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    auto env = cub_bench_env(alloc, launch);
    _CCCL_TRY_RUNTIME_API(
      cub::DeviceSelect::If,
      "DeviceSelect::If failed",
      input,
      output.begin(),
      selected_count.begin(),
      static_cast<::cuda::std::int64_t>(elements),
      synthesizing_select_even_t{},
      env);
  });
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

NVBENCH_BENCH_TYPES(select_if, NVBENCH_TYPE_AXES(synthesizing_input_types))
  .set_name("base")
  .set_type_axes_names({"Input{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
