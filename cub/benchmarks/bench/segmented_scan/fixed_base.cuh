// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/device/device_segmented_scan.cuh>

#include <cuda/std/cstdint>

#include <nvbench_helper.cuh>

#include "common.cuh"

template <typename T>
static void fixed_size_segmented_scan(nvbench::state& state, nvbench::type_list<T>)
{
#if !TUNE_BASE
  using policy_t = policy_selector_t<TUNE_THREADS, TUNE_ITEMS, TUNE_MAX_SEGMENTS_PER_BLOCK>;
#endif

  const auto segment_size = static_cast<int>(state.get_int64("SegmentSize{io}"));
  // Only whole segments, since all segments have the same size
  const auto num_segments = static_cast<::cuda::std::int64_t>(state.get_int64("Elements{io}") / segment_size);
  const auto elements     = static_cast<size_t>(num_segments * segment_size);
  auto& summary           = state.add_summary("user/derived/segment_count");
  summary.set_string("name", "#Segments");
  summary.set_int64("value", num_segments);

  thrust::device_vector<T> input = generate(elements);
  thrust::device_vector<T> output(elements, thrust::default_init);

  const T* d_input = thrust::raw_pointer_cast(input.data());
  T* d_output      = thrust::raw_pointer_cast(output.data());

  state.add_element_count(elements, "Elements");
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_writes<T>(elements);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    auto env = cub_bench_env(
      alloc,
      launch
#if !TUNE_BASE
      ,
      cuda::execution::tune(policy_t{})
#endif // !TUNE_BASE
    );
    _CCCL_TRY_RUNTIME_API(
      cub::DeviceSegmentedScan::ExclusiveSegmentedScan,
      "ExclusiveSegmentedScan failed",
      d_input,
      d_output,
      num_segments,
      segment_size,
      op_t{},
      T{},
      env);
  });
}

NVBENCH_BENCH_TYPES(fixed_size_segmented_scan, NVBENCH_TYPE_AXES(benched_value_types))
  .set_name("fixed_size_segments")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(18, 26, 4))
  .add_int64_axis("SegmentSize{io}", {1, 7, 32, 51, 123, 233, 513, 1337, 4417});
