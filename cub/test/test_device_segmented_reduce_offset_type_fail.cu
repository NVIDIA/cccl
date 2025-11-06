// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

// %PARAM% TEST_ERR err 0:1:2:3:4:5:6:7

#include <cub/device/device_segmented_reduce.cuh>

template <typename T>
void mark_as_used(T&&)
{}

int main()
{
  using offset_t = float; // error
  // using offset_t = int; // ok
  float *d_in{}, *d_out{};
  cub::KeyValuePair<float, float>* d_kv_out{};
  ::cuda::std::pair<float, float>* d_pair_out{};
  offset_t* d_offsets{};
  std::size_t temp_storage_bytes{};
  std::uint8_t* d_temp_storage{};

  // Only one of these is used per path, suppress undesired diagnostics:
  mark_as_used(d_out);
  mark_as_used(d_kv_out);
  mark_as_used(d_pair_out);
  mark_as_used(d_offsets);

#if TEST_ERR == 0
  // expected-error-0 {{"Offset iterator value type should be integral."}}
  cub::DeviceSegmentedReduce::Reduce(
    d_temp_storage, temp_storage_bytes, d_in, d_out, 0, d_offsets, d_offsets + 1, ::cuda::minimum<>{}, 0);
#elif TEST_ERR == 1
  // expected-error-1 {{"Offset iterator value type should be integral."}}
  cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, 0, d_offsets, d_offsets + 1);
#elif TEST_ERR == 2
  // expected-error-2 {{"Offset iterator value type should be integral."}}
  cub::DeviceSegmentedReduce::Min(d_temp_storage, temp_storage_bytes, d_in, d_out, 0, d_offsets, d_offsets + 1);
#elif TEST_ERR == 3
  // expected-error-3 {{"Offset iterator value type should be integral."}}
  cub::DeviceSegmentedReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, 0, d_offsets, d_offsets + 1);
#elif TEST_ERR == 4
  // expected-error-4 {{"Output key type must be int."}}
  cub::DeviceSegmentedReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_in, d_kv_out, 0, d_offsets, d_offsets + 1);
#elif TEST_ERR == 5
  // expected-error-5 {{"Output key type must be int."}}
  cub::DeviceSegmentedReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_in, d_pair_out, 0, 1);
#elif TEST_ERR == 6
  // expected-error-6 {{"Output key type must be int."}}
  cub::DeviceSegmentedReduce::ArgMax(d_temp_storage, temp_storage_bytes, d_in, d_kv_out, 0, d_offsets, d_offsets + 1);
#elif TEST_ERR == 7
  // expected-error-7 {{"Output key type must be int."}}
  cub::DeviceSegmentedReduce::ArgMax(d_temp_storage, temp_storage_bytes, d_in, d_pair_out, 0, 1);
#endif
}
