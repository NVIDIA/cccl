// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cub/device/device_reduce.cuh>

struct invalid_env
{
  using property_keys = cuda::execution::property_key_list<cuda::get_stream_t>;

  cuda::stream_ref query(cuda::get_stream_t) noexcept
  {
    return cuda::stream_ref{cudaStream_t{}};
  }
};

int main()
{
  int* ptr{};

  // expected-error {{"The environment advertises a query expression that is not valid for a const environment."}}
  auto error = cub::DeviceReduce::Sum(ptr, ptr, 0, invalid_env{});
  if (error != cudaSuccess)
  {
    return 1;
  }
}
