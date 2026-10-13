// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define CCCL_IGNORE_DEPRECATED_API

#include <cub/device/device_scan.cuh>

#include <nvbench_helper.cuh>

#include "iterators.cuh"

// Scan-by-key loads keys and values through one BlockLoad algorithm. The default policy, and most of the tuned
// policies, request a warp-transpose load. Both iterators here are synthesizing, so neither load reads memory and the
// BlockLoad change skips that exchange. Keys are the item index divided by a fixed segment size, so each run has that
// many values except the last.
inline constexpr ::cuda::std::int64_t synthesizing_scan_by_key_segment_size = 4096;

struct synthesizing_scan_by_key_key_t
{
  _CCCL_HOST_DEVICE constexpr ::cuda::std::int64_t operator()(::cuda::std::int64_t index) const noexcept
  {
    return index / synthesizing_scan_by_key_segment_size;
  }
};

template <class Input>
void scan_by_key(nvbench::state& state, nvbench::type_list<Input>)
try
{
  const auto elements = static_cast<::cuda::std::int64_t>(state.get_int64("Elements{io}"));
  auto values         = Input::make(static_cast<::cuda::std::int64_t>(elements));
  using value_t       = cub::detail::it_value_t<decltype(values)>;
  auto output         = synthesizing_make_output<value_t>(static_cast<std::size_t>(elements));
  const auto keys =
    cuda::transform_iterator{cuda::counting_iterator<::cuda::std::int64_t>{0}, synthesizing_scan_by_key_key_t{}};

  state.add_element_count(elements);
  state.add_global_memory_writes<value_t>(elements);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    auto env = cub_bench_env(alloc, launch);
    if constexpr (Input::scalar)
    {
      _CCCL_TRY_RUNTIME_API(
        cub::DeviceScan::InclusiveScanByKey,
        "InclusiveScanByKey failed",
        keys,
        values,
        output.begin(),
        ::cuda::std::plus<>{},
        elements,
        ::cuda::std::equal_to<>{},
        env);
    }
    else
    {
      _CCCL_TRY_RUNTIME_API(
        cub::DeviceScan::InclusiveScanByKey,
        "InclusiveScanByKey failed",
        keys,
        values,
        output.begin(),
        synthesizing_tuple_plus_t<typename Input::value_type>{},
        elements,
        ::cuda::std::equal_to<>{},
        env);
    }
  });
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

NVBENCH_BENCH_TYPES(scan_by_key, NVBENCH_TYPE_AXES(synthesizing_input_types))
  .set_name("base")
  .set_type_axes_names({"Input{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
