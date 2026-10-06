// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#define CCCL_IGNORE_DEPRECATED_API

#include <cub/device/device_scan.cuh>

#include <nvbench_helper.cuh>

#include "iterators.cuh"

// Inclusive sum of a synthesizing input. Nothing is read from global memory; the only traffic is the output and the
// scan's temporary storage. Scalar inputs use InclusiveSum so they take the tuned plus policy. Zip inputs need a
// component-wise operator, so they take the generic-operator policy.
template <class Input>
void scan(nvbench::state& state, nvbench::type_list<Input>)
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
        cub::DeviceScan::InclusiveSum,
        "InclusiveSum failed",
        input,
        output.begin(),
        static_cast<::cuda::std::int64_t>(elements),
        env);
    }
    else
    {
      _CCCL_TRY_RUNTIME_API(
        cub::DeviceScan::InclusiveScan,
        "InclusiveScan failed",
        input,
        output.begin(),
        synthesizing_tuple_plus_t<typename Input::value_type>{},
        static_cast<::cuda::std::int64_t>(elements),
        env);
    }
  });
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

NVBENCH_BENCH_TYPES(scan, NVBENCH_TYPE_AXES(synthesizing_input_types))
  .set_name("base")
  .set_type_axes_names({"Input{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
