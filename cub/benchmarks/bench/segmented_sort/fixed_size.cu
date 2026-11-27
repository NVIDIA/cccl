// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_segmented_sort.cuh>

#include <thrust/host_vector.h>

#include <cuda/argument>

#include <nvbench_helper.cuh>

template <class T>
void fixed_size(nvbench::state& state, nvbench::type_list<T>)
{
  const auto elements     = state.get_int64("Elements{io}");
  const int segment_size  = static_cast<int>(state.get_int64("SegmentSize"));
  const auto num_segments = elements / segment_size;
  const auto api          = state.get_string("API");

  thrust::device_vector<T> input = generate(elements, bit_entropy::_1_000);
  thrust::device_vector<T> output(elements, thrust::no_init);
  thrust::host_vector<cuda::std::int64_t> host_offsets(num_segments + 1);
  for (cuda::std::int64_t i = 0; i <= num_segments; ++i)
  {
    host_offsets[i] = i * segment_size;
  }
  thrust::device_vector<cuda::std::int64_t> offsets = host_offsets;
  const T* in                                       = thrust::raw_pointer_cast(input.data());
  T* out                                            = thrust::raw_pointer_cast(output.data());
  const auto begin                                  = thrust::raw_pointer_cast(offsets.data());

  const auto sort = [&](void* temp, size_t& bytes, cudaStream_t stream) {
    if (api == "offsets")
    {
      return cub::DeviceSegmentedSort::SortKeys(temp, bytes, in, out, elements, num_segments, begin, begin + 1, stream);
    }
    return cub::DeviceSegmentedSort::SortKeys(
      temp, bytes, in, out, cuda::args::immediate{segment_size}, num_segments, stream);
  };
  size_t bytes{};
  _CCCL_TRY_RUNTIME_API(sort, "SortKeys storage query failed", nullptr, bytes, cudaStream_t{});
  thrust::device_vector<cuda::std::uint8_t> storage(bytes, thrust::no_init);

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_writes<T>(elements);
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) {
               _CCCL_TRY_RUNTIME_API(
                 sort, "SortKeys failed", thrust::raw_pointer_cast(storage.data()), bytes, launch.get_stream());
               _CCCL_TRY_RUNTIME_API(cudaStreamSynchronize, "Stream synchronization failed", launch.get_stream());
             });
}

NVBENCH_BENCH_TYPES(fixed_size, NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::int32_t, nvbench::float32_t>))
  .set_name("fixed_size")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", {22, 24})
  .add_int64_axis("SegmentSize", {16, 32, 128, 512, 4096, 16384})
  .add_string_axis("API", {"offsets", "fixed"});
