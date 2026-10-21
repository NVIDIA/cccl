// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_reduce.cuh>

#include <cuda/std/cstdint>

#include <cstddef>

#include <nvbench_helper.cuh>

template <typename OffsetT>
void sum_small(nvbench::state& state, nvbench::type_list<OffsetT>)
{
  using value_t                        = cuda::std::uint32_t;
  const auto count                     = static_cast<OffsetT>(state.get_int64("Elements{io}"));
  const auto offset                    = state.get_int64("Offset{io}");
  thrust::device_vector<value_t> input = generate(count + offset);
  thrust::device_vector<value_t> output(1);
  auto* d_input  = thrust::raw_pointer_cast(input.data()) + offset;
  auto* d_output = thrust::raw_pointer_cast(output.data());

  std::size_t bytes = 0;
  _CCCL_TRY_RUNTIME_API(cub::DeviceReduce::Sum, "Sum storage query failed", nullptr, bytes, d_input, d_output, count);
  thrust::device_vector<char> storage(bytes);
  state.add_element_count(count);
  state.add_global_memory_reads<value_t>(count);
  state.add_global_memory_writes<value_t>(1);
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    _CCCL_TRY_RUNTIME_API(
      cub::DeviceReduce::Sum, "Sum failed", storage.data().get(), bytes, d_input, d_output, count, launch.get_stream());
  });
}

NVBENCH_BENCH_TYPES(sum_small, NVBENCH_TYPE_AXES(nvbench::type_list<cuda::std::int32_t, cuda::std::int64_t>))
  .set_name("base")
  .set_type_axes_names({"OffsetT{ct}"})
  .add_int64_axis("Elements{io}", {4095, 4096, 4097, 6143, 6144, 8191, 8192, 8193, 16385})
  .add_int64_axis("Offset{io}", {0, 1});
