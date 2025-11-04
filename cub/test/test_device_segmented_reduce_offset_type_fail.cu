// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

// %PARAM% TEST_ERR err 0:1:2:3:4:5

#include <cub/device/device_segmented_reduce.cuh>

int main()
{
  using offset_t = float; // error
  // using offset_t = int; // ok
  float *d_in{}, *d_out{};
  offset_t* d_offsets{};
  std::size_t temp_storage_bytes{};
  std::uint8_t* d_temp_storage{};

#if TEST_ERR == 0
  // expected-error {{"Offset iterator type should be integral."}}
  cub::DeviceSegmentedReduce::Reduce(
    d_temp_storage, temp_storage_bytes, d_in, d_out, 0, d_offsets, d_offsets + 1, ::cuda::minimum<>{}, 0);

#elif TEST_ERR == 1
  // expected-error {{"Offset iterator type should be integral."}}
  cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, 0, d_offsets, d_offsets + 1);

#elif TEST_ERR == 2
  // expected-error {{"Offset iterator type should be integral."}}
  cub::DeviceSegmentedReduce::Min(d_temp_storage, temp_storage_bytes, d_in, d_out, 0, d_offsets, d_offsets + 1);

#elif TEST_ERR == 3
  // expected-error {{"Offset iterator type should be integral."}}
  cub::DeviceSegmentedReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_in, d_out, 0, d_offsets, d_offsets + 1);

#elif TEST_ERR == 4
  // expected-error {{"Offset iterator type should be integral."}}
  cub::DeviceSegmentedReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, 0, d_offsets, d_offsets + 1);

#elif TEST_ERR == 5
  // expected-error {{"Offset iterator type should be integral."}}
  cub::DeviceSegmentedReduce::ArgMax(d_temp_storage, temp_storage_bytes, d_in, d_out, 0, d_offsets, d_offsets + 1);
#endif
}
