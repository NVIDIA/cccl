// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#define CCCL_IGNORE_DEPRECATED_API

#include <cub/device/device_segmented_scan.cuh>

#include <nvbench_helper.cuh>

#include "iterators.cuh"

// Segmented scan always loads values with a warp-transpose BlockLoad (timesliced for a large accumulator). A
// synthesizing input never reads memory, so that exchange is the cost the BlockLoad change removes. Offsets are a
// transform of a counting iterator, so they are not read from memory either. Each segment covers a fixed number of
// items and the last segment holds the remainder.
inline constexpr ::cuda::std::int64_t synthesizing_segment_size = 4096;

struct synthesizing_segment_begin_t
{
  _CCCL_HOST_DEVICE constexpr ::cuda::std::int64_t operator()(::cuda::std::int64_t segment) const noexcept
  {
    return segment * synthesizing_segment_size;
  }
};

struct synthesizing_segment_end_t
{
  ::cuda::std::int64_t num_items;

  _CCCL_HOST_DEVICE constexpr ::cuda::std::int64_t operator()(::cuda::std::int64_t segment) const noexcept
  {
    const auto end = (segment + 1) * synthesizing_segment_size;
    return end < num_items ? end : num_items;
  }
};

template <class Input>
void segmented_scan(nvbench::state& state, nvbench::type_list<Input>)
try
{
  const auto elements     = static_cast<::cuda::std::int64_t>(state.get_int64("Elements{io}"));
  const auto num_segments = (elements + synthesizing_segment_size - 1) / synthesizing_segment_size;
  auto input              = Input::make(elements);
  using value_t           = cub::detail::it_value_t<decltype(input)>;
  auto output             = synthesizing_make_output<value_t>(static_cast<std::size_t>(elements));
  const auto begin_offsets =
    cuda::transform_iterator{cuda::counting_iterator<::cuda::std::int64_t>{0}, synthesizing_segment_begin_t{}};
  const auto end_offsets =
    cuda::transform_iterator{cuda::counting_iterator<::cuda::std::int64_t>{0}, synthesizing_segment_end_t{elements}};

  state.add_element_count(elements);
  state.add_global_memory_writes<value_t>(elements);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    auto env = cub_bench_env(alloc, launch);
    if constexpr (Input::scalar)
    {
      _CCCL_TRY_RUNTIME_API(
        cub::DeviceSegmentedScan::InclusiveSegmentedSum,
        "InclusiveSegmentedSum failed",
        input,
        output.begin(),
        begin_offsets,
        end_offsets,
        num_segments,
        env);
    }
    else
    {
      _CCCL_TRY_RUNTIME_API(
        cub::DeviceSegmentedScan::InclusiveSegmentedScan,
        "InclusiveSegmentedScan failed",
        input,
        output.begin(),
        begin_offsets,
        end_offsets,
        num_segments,
        synthesizing_tuple_plus_t<typename Input::value_type>{},
        env);
    }
  });
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

NVBENCH_BENCH_TYPES(segmented_scan, NVBENCH_TYPE_AXES(synthesizing_input_types))
  .set_name("base")
  .set_type_axes_names({"Input{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
