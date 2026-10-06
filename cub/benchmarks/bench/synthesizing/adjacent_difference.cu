// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#define CCCL_IGNORE_DEPRECATED_API

#include <cub/device/device_adjacent_difference.cuh>

#include <nvbench_helper.cuh>

#include "iterators.cuh"

// Left adjacent difference of a synthesizing input. Scalar inputs use the default subtraction. Zip inputs subtract
// each tuple element.
template <class Input>
void subtract_left(nvbench::state& state, nvbench::type_list<Input>)
try
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  auto input          = Input::make();
  using value_t       = cub::detail::it_value_t<decltype(input)>;
  auto output         = synthesizing_make_output<value_t>(elements);

  state.add_element_count(elements);
  state.add_global_memory_writes<value_t>(elements);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    auto env = cub_bench_env(alloc, launch);
    if constexpr (Input::scalar)
    {
      _CCCL_TRY_RUNTIME_API(
        cub::DeviceAdjacentDifference::SubtractLeftCopy,
        "SubtractLeftCopy failed",
        input,
        output.begin(),
        static_cast<::cuda::std::int64_t>(elements),
        ::cuda::std::minus<>{},
        env);
    }
    else
    {
      _CCCL_TRY_RUNTIME_API(
        cub::DeviceAdjacentDifference::SubtractLeftCopy,
        "SubtractLeftCopy failed",
        input,
        output.begin(),
        static_cast<::cuda::std::int64_t>(elements),
        synthesizing_tuple_minus_t<typename Input::value_type>{},
        env);
    }
  });
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

NVBENCH_BENCH_TYPES(subtract_left, NVBENCH_TYPE_AXES(synthesizing_input_types))
  .set_name("base")
  .set_type_axes_names({"Input{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
