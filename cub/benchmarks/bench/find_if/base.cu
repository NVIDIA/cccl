// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_find_if.cuh>

#include <thrust/count.h>
#include <thrust/detail/internal_functional.h>
#include <thrust/find.h>

#include <nvbench_helper.cuh>

template <typename T>
void find_if(nvbench::state& state, nvbench::type_list<T>)
{
  T val = 1;
  // set up input
  const auto elements       = state.get_int64("Elements");
  const auto common_prefix  = state.get_float64("MismatchAt");
  const auto mismatch_point = elements * common_prefix;

  thrust::device_vector<T> dinput(elements, thrust::no_init);
  thrust::fill(dinput.begin(), dinput.begin() + mismatch_point, 0);
  thrust::fill(dinput.begin() + mismatch_point, dinput.end(), val);
  thrust::device_vector<T> d_result(1, thrust::no_init);
  ///

  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes{};

  cub::DeviceFind::FindIf(
    d_temp_storage,
    temp_storage_bytes,
    thrust::raw_pointer_cast(dinput.data()),
    thrust::raw_pointer_cast(d_result.data()),
    thrust::detail::equal_to_value<T>(val),
    dinput.size(),
    0);

  thrust::device_vector<uint8_t> temp_storage(temp_storage_bytes, thrust::no_init);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  state.exec(nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    cub::DeviceFind::FindIf(
      d_temp_storage,
      temp_storage_bytes,
      thrust::raw_pointer_cast(dinput.data()),
      thrust::raw_pointer_cast(d_result.data()),
      thrust::detail::equal_to_value<T>(val),
      dinput.size(),
      launch.get_stream());
  });
}

NVBENCH_BENCH_TYPES(find_if, NVBENCH_TYPE_AXES(fundamental_types))
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4))
  .add_float64_axis("MismatchAt", std::vector{1.0, 0.5, 0.0});
