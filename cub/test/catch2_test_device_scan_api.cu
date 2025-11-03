// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <cub/device/device_scan.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <c2h/catch2_test_helper.h>

C2H_TEST("Device inclusive scan works", "[scan][device]")
{
  // example-begin device-inclusive-scan
  thrust::device_vector<int> input{0, -1, 2, -3, 4, -5, 6};
  thrust::device_vector<int> out(input.size());

  int init = 1;
  size_t temp_storage_bytes{};

  cub::DeviceScan::InclusiveScanInit(
    nullptr, temp_storage_bytes, input.begin(), out.begin(), cuda::maximum<>{}, init, static_cast<int>(input.size()));

  // Allocate temporary storage for inclusive scan
  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);

  // Run inclusive prefix sum
  cub::DeviceScan::InclusiveScanInit(
    thrust::raw_pointer_cast(temp_storage.data()),
    temp_storage_bytes,
    input.begin(),
    out.begin(),
    cuda::maximum<>{},
    init,
    static_cast<int>(input.size()));

  thrust::host_vector<int> expected{1, 1, 2, 2, 4, 4, 6};
  // example-end device-inclusive-scan

  REQUIRE(expected == out);
}
