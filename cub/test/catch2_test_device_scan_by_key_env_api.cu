// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_scan.cuh>

#include <thrust/device_vector.h>

#include <cuda/devices>
#include <cuda/stream>

#include <iostream>

#include <c2h/catch2_test_helper.h>

C2H_TEST("cub::DeviceScan::ExclusiveSumByKey accepts stream environment", "[scan][by_key][env]")
{
  // example-begin exclusive-sum-by-key-env
  auto keys   = thrust::device_vector<int>{0, 0, 1, 1, 1, 2, 2};
  auto input  = thrust::device_vector<float>{8.0f, 6.0f, 7.0f, 5.0f, 3.0f, 0.0f, 9.0f};
  auto output = thrust::device_vector<float>(7);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceScan::ExclusiveSumByKey(
    keys.begin(), input.begin(), output.begin(), static_cast<int>(input.size()), cuda::std::equal_to<>{}, stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceScan::ExclusiveSumByKey failed with status: " << error << '\n';
  }

  thrust::device_vector<float> expected{0.0f, 8.0f, 0.0f, 7.0f, 12.0f, 0.0f, 0.0f};
  // example-end exclusive-sum-by-key-env

  stream.sync();
  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

C2H_TEST("cub::DeviceScan::ExclusiveScanByKey accepts stream environment", "[scan][by_key][env]")
{
  // example-begin exclusive-scan-by-key-env
  auto op     = cuda::std::plus{};
  auto keys   = thrust::device_vector<int>{0, 0, 1, 1, 1, 2, 2};
  auto input  = thrust::device_vector<float>{8.0f, 6.0f, 7.0f, 5.0f, 3.0f, 0.0f, 9.0f};
  auto output = thrust::device_vector<float>(7);
  auto init   = 0.0f;

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceScan::ExclusiveScanByKey(
    keys.begin(),
    input.begin(),
    output.begin(),
    op,
    init,
    static_cast<int>(input.size()),
    cuda::std::equal_to<>{},
    stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceScan::ExclusiveScanByKey failed with status: " << error << '\n';
  }

  thrust::device_vector<float> expected{0.0f, 8.0f, 0.0f, 7.0f, 12.0f, 0.0f, 0.0f};
  // example-end exclusive-scan-by-key-env

  stream.sync();
  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

C2H_TEST("cub::DeviceScan::InclusiveSumByKey accepts stream environment", "[scan][by_key][env]")
{
  // example-begin inclusive-sum-by-key-env
  auto keys   = thrust::device_vector<int>{0, 0, 1, 1, 1, 2, 2};
  auto input  = thrust::device_vector<float>{8.0f, 6.0f, 7.0f, 5.0f, 3.0f, 0.0f, 9.0f};
  auto output = thrust::device_vector<float>(7);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceScan::InclusiveSumByKey(
    keys.begin(), input.begin(), output.begin(), static_cast<int>(input.size()), cuda::std::equal_to<>{}, stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceScan::InclusiveSumByKey failed with status: " << error << '\n';
  }

  thrust::device_vector<float> expected{8.0f, 14.0f, 7.0f, 12.0f, 15.0f, 0.0f, 9.0f};
  // example-end inclusive-sum-by-key-env

  stream.sync();
  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

C2H_TEST("cub::DeviceScan::InclusiveScanByKey accepts stream environment", "[scan][by_key][env]")
{
  // example-begin inclusive-scan-by-key-env
  auto op     = cuda::std::plus{};
  auto keys   = thrust::device_vector<int>{0, 0, 1, 1, 1, 2, 2};
  auto input  = thrust::device_vector<float>{8.0f, 6.0f, 7.0f, 5.0f, 3.0f, 0.0f, 9.0f};
  auto output = thrust::device_vector<float>(7);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceScan::InclusiveScanByKey(
    keys.begin(),
    input.begin(),
    output.begin(),
    op,
    static_cast<int>(input.size()),
    cuda::std::equal_to<>{},
    stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceScan::InclusiveScanByKey failed with status: " << error << '\n';
  }

  thrust::device_vector<float> expected{8.0f, 14.0f, 7.0f, 12.0f, 15.0f, 0.0f, 9.0f};
  // example-end inclusive-scan-by-key-env

  stream.sync();
  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

#if _CCCL_STD_VER >= 2020

// example-begin exclusive-sum-by-key-policy-selector
struct ScanByKeyPolicySelector
{
  __host__ __device__ constexpr auto operator()(cuda::compute_capability cc) const -> cub::ScanByKeyPolicy
  {
    return {.threads_per_block = 256,
            .items_per_thread  = cc > cuda::compute_capability{9, 0} ? 15 : 12,
            .load_algorithm    = cub::BLOCK_LOAD_WARP_TRANSPOSE,
            .load_modifier     = cub::LOAD_DEFAULT,
            .store_algorithm   = cub::BLOCK_STORE_WARP_TRANSPOSE,
            .scan_algorithm    = cub::BLOCK_SCAN_WARP_SCANS,
            .lookback_delay    = cub::LookbackDelayPolicy{cub::LookbackDelayAlgorithm::fixed_delay, 832, 1165}};
  }
};
// example-end exclusive-sum-by-key-policy-selector

C2H_TEST("cub::DeviceScan::ExclusiveSumByKey env-based API with tuning", "[scan][by_key][env]")
{
  // example-begin exclusive-sum-by-key-tuning
  auto keys   = thrust::device_vector<int>{0, 0, 1, 1, 1, 2, 2};
  auto input  = thrust::device_vector<float>{8.0f, 6.0f, 7.0f, 5.0f, 3.0f, 0.0f, 9.0f};
  auto output = thrust::device_vector<float>(7, thrust::no_init);

  const auto error = cub::DeviceScan::ExclusiveSumByKey(
    keys.begin(),
    input.begin(),
    output.begin(),
    static_cast<int>(input.size()),
    cuda::std::equal_to<>{},
    cuda::execution::tune(ScanByKeyPolicySelector{}));
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceScan::ExclusiveSumByKey failed with status: " << error << '\n';
  }

  thrust::device_vector<float> expected{0.0f, 8.0f, 0.0f, 7.0f, 12.0f, 0.0f, 0.0f};
  // example-end exclusive-sum-by-key-tuning

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

#endif // _CCCL_STD_VER >= 2020
