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

C2H_TEST("cub::DeviceScan::ExclusiveSum non-env overload is not ambiguous", "[scan][device]")
{
  thrust::device_vector<int> input(1);
  thrust::device_vector<int> out(1);
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, input.begin(), out.begin(), 1);
}

C2H_TEST("cub::DeviceScan::ExclusiveSum non-env in-place overload is not ambiguous", "[scan][device]")
{
  thrust::device_vector<int> data(1);
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, data.begin(), 1);
}

C2H_TEST("cub::DeviceScan::ExclusiveScan non-env overload is not ambiguous", "[scan][device]")
{
  thrust::device_vector<int> input(1);
  thrust::device_vector<int> out(1);
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveScan(nullptr, temp_storage_bytes, input.begin(), out.begin(), cuda::std::plus<>{}, 5, 1);
}

C2H_TEST("cub::DeviceScan::ExclusiveScan non-env in-place overload is not ambiguous", "[scan][device]")
{
  thrust::device_vector<int> data(1);
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveScan(nullptr, temp_storage_bytes, data.begin(), cuda::std::plus<>{}, 5, 1);
}

C2H_TEST("cub::DeviceScan::ExclusiveScan FutureValue non-env overload is not ambiguous", "[scan][device]")
{
  thrust::device_vector<int> input(1);
  thrust::device_vector<int> out(1);
  thrust::device_vector<int> init_storage(1, 5);
  auto future_init          = cub::FutureValue<int>(thrust::raw_pointer_cast(init_storage.data()));
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveScan(
    nullptr, temp_storage_bytes, input.begin(), out.begin(), cuda::std::plus<>{}, future_init, 1);
}

C2H_TEST("cub::DeviceScan::ExclusiveScan FutureValue non-env in-place overload is not ambiguous", "[scan][device]")
{
  thrust::device_vector<int> data(1);
  thrust::device_vector<int> init_storage(1, 5);
  auto future_init          = cub::FutureValue<int>(thrust::raw_pointer_cast(init_storage.data()));
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveScan(nullptr, temp_storage_bytes, data.begin(), cuda::std::plus<>{}, future_init, 1);
}

C2H_TEST("cub::DeviceScan::InclusiveSum non-env overload is not ambiguous", "[scan][device]")
{
  thrust::device_vector<int> input(1);
  thrust::device_vector<int> out(1);
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes, input.begin(), out.begin(), 1);
}

C2H_TEST("cub::DeviceScan::InclusiveSum non-env in-place overload is not ambiguous", "[scan][device]")
{
  thrust::device_vector<int> data(1);
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes, data.begin(), 1);
}

C2H_TEST("cub::DeviceScan::InclusiveScan non-env overload is not ambiguous", "[scan][device]")
{
  thrust::device_vector<int> input(1);
  thrust::device_vector<int> out(1);
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::InclusiveScan(nullptr, temp_storage_bytes, input.begin(), out.begin(), cuda::std::plus<>{}, 1);
}

C2H_TEST("cub::DeviceScan::InclusiveScan non-env in-place overload is not ambiguous", "[scan][device]")
{
  thrust::device_vector<int> data(1);
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::InclusiveScan(nullptr, temp_storage_bytes, data.begin(), cuda::std::plus<>{}, 1);
}

C2H_TEST("cub::DeviceScan::InclusiveScanInit non-env overload is not ambiguous", "[scan][device]")
{
  thrust::device_vector<int> input(1);
  thrust::device_vector<int> out(1);
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::InclusiveScanInit(nullptr, temp_storage_bytes, input.begin(), out.begin(), cuda::std::plus<>{}, 5, 1);
}

C2H_TEST("cub::DeviceScan::ExclusiveSumByKey non-env overload is not ambiguous", "[scan][device]")
{
  thrust::device_vector<int> keys(1);
  thrust::device_vector<int> values_in(1);
  thrust::device_vector<int> values_out(1);
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSumByKey(
    nullptr, temp_storage_bytes, keys.begin(), values_in.begin(), values_out.begin(), 1);
}

C2H_TEST("cub::DeviceScan::ExclusiveScanByKey non-env overload is not ambiguous", "[scan][device]")
{
  thrust::device_vector<int> keys(1);
  thrust::device_vector<int> values_in(1);
  thrust::device_vector<int> values_out(1);
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveScanByKey(
    nullptr, temp_storage_bytes, keys.begin(), values_in.begin(), values_out.begin(), cuda::std::plus<>{}, 5, 1);
}

C2H_TEST("cub::DeviceScan::InclusiveSumByKey non-env overload is not ambiguous", "[scan][device]")
{
  thrust::device_vector<int> keys(1);
  thrust::device_vector<int> values_in(1);
  thrust::device_vector<int> values_out(1);
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::InclusiveSumByKey(
    nullptr, temp_storage_bytes, keys.begin(), values_in.begin(), values_out.begin(), 1);
}

C2H_TEST("cub::DeviceScan::InclusiveScanByKey non-env overload is not ambiguous", "[scan][device]")
{
  thrust::device_vector<int> keys(1);
  thrust::device_vector<int> values_in(1);
  thrust::device_vector<int> values_out(1);
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::InclusiveScanByKey(
    nullptr, temp_storage_bytes, keys.begin(), values_in.begin(), values_out.begin(), cuda::std::plus<>{}, 1);
}
