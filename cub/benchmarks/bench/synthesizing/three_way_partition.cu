// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define CCCL_IGNORE_DEPRECATED_API

#include <cub/device/device_partition.cuh>

#include <nvbench_helper.cuh>

#include "iterators.cuh"

// Three-way partition's tuned policy requests a warp-transpose BlockLoad for 16-bit and wider values. The input is
// synthesizing, so that load does not read memory. Items are split by value modulo 3. A zip uses its first element.
struct synthesizing_mod3_t
{
  int remainder;

  template <class T>
  _CCCL_HOST_DEVICE constexpr bool operator()(const T& value) const noexcept
  {
    if constexpr (::cuda::std::is_integral_v<T>)
    {
      return static_cast<int>(value % T{3}) == remainder;
    }
    else
    {
      return (*this)(::cuda::std::get<0>(value));
    }
  }
};

template <class Input>
void three_way_partition(nvbench::state& state, nvbench::type_list<Input>)
try
{
  const auto elements = static_cast<::cuda::std::int64_t>(state.get_int64("Elements{io}"));
  auto input          = Input::make(elements);
  using value_t       = cub::detail::it_value_t<decltype(input)>;
  auto first          = synthesizing_make_output<value_t>(static_cast<std::size_t>(elements));
  auto second         = synthesizing_make_output<value_t>(static_cast<std::size_t>(elements));
  auto unselected     = synthesizing_make_output<value_t>(static_cast<std::size_t>(elements));
  thrust::device_vector<::cuda::std::int64_t> num_selected(2, thrust::no_init);

  state.add_element_count(elements);
  state.add_global_memory_writes<value_t>(elements);
  state.add_global_memory_writes<::cuda::std::int64_t>(2);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    auto env = cub_bench_env(alloc, launch);
    _CCCL_TRY_RUNTIME_API(
      cub::DevicePartition::If,
      "DevicePartition::If failed",
      input,
      first.begin(),
      second.begin(),
      unselected.begin(),
      num_selected.begin(),
      elements,
      synthesizing_mod3_t{0},
      synthesizing_mod3_t{1},
      env);
  });
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

NVBENCH_BENCH_TYPES(three_way_partition, NVBENCH_TYPE_AXES(synthesizing_input_types))
  .set_name("base")
  .set_type_axes_names({"Input{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
