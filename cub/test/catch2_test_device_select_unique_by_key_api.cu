// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_select.cuh>

#include <thrust/device_vector.h>

#include <cuda/__execution/tune.h>
#include <cuda/devices>
#include <cuda/std/cstdint>
#include <cuda/stream>

#include <cstddef>
#include <iostream>

#include "cub_test_macros.h"

CUB_TEST("DeviceSelect::UniqueByKey legacy size-query is unambiguous", "[select][device]", CUB_SMALL)
{
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  int* d_keys_in            = nullptr;
  int* d_values_in          = nullptr;
  int* d_keys_out           = nullptr;
  int* d_values_out         = nullptr;
  int* d_num_selected       = nullptr;
  const int n               = 0;

  REQUIRE(cudaSuccess
          == cub::DeviceSelect::UniqueByKey(
            d_temp_storage, temp_storage_bytes, d_keys_in, d_values_in, d_keys_out, d_values_out, d_num_selected, n));
}

CUB_TEST("cub::DeviceSelect::UniqueByKey accepts env with stream", "[select][env]", CUB_SMALL)
{
  // example-begin select-uniquebykey-env
  auto keys_in      = thrust::device_vector<int>{0, 2, 2, 9, 5, 5, 5, 8};
  auto values_in    = thrust::device_vector<int>{1, 2, 3, 4, 5, 6, 7, 8};
  auto keys_out     = thrust::device_vector<int>(5);
  auto values_out   = thrust::device_vector<int>(5);
  auto num_selected = thrust::device_vector<int>(1);

  const cuda::stream stream{cuda::devices[0]};
  const cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceSelect::UniqueByKey(
    keys_in.begin(),
    values_in.begin(),
    keys_out.begin(),
    values_out.begin(),
    num_selected.begin(),
    keys_in.size(),
    cuda::std::equal_to<>{},
    stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSelect::UniqueByKey failed with status: " << error << '\n';
  }

  const thrust::device_vector<int> expected_keys{0, 2, 9, 5, 8};
  const thrust::device_vector<int> expected_values{1, 2, 4, 5, 8};
  const thrust::device_vector<int> expected_num_selected{5};
  // example-end select-uniquebykey-env

  stream.sync();
  REQUIRE(error == cudaSuccess);
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
  REQUIRE(num_selected == expected_num_selected);
}

CUB_TEST("cub::DeviceSelect::UniqueByKey accepts env with stream without equality_op", "[select][env]", CUB_SMALL)
{
  // example-begin select-uniquebykey-default-eq-env
  // Same setup/expectations as the explicit equality_op test above, but relying on default equality.
  auto keys_in      = thrust::device_vector<int>{0, 2, 2, 9, 5, 5, 5, 8};
  auto values_in    = thrust::device_vector<int>{1, 2, 3, 4, 5, 6, 7, 8};
  auto keys_out     = thrust::device_vector<int>(5);
  auto values_out   = thrust::device_vector<int>(5);
  auto num_selected = thrust::device_vector<int>(1);

  const cuda::stream stream{cuda::devices[0]};
  const cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceSelect::UniqueByKey(
    keys_in.begin(),
    values_in.begin(),
    keys_out.begin(),
    values_out.begin(),
    num_selected.begin(),
    keys_in.size(),
    stream_ref);

  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSelect::UniqueByKey without equality_op failed with status: " << error << '\n';
  }

  const thrust::device_vector<int> expected_keys{0, 2, 9, 5, 8};
  const thrust::device_vector<int> expected_values{1, 2, 4, 5, 8};
  const thrust::device_vector<int> expected_num_selected{5};
  // example-end select-uniquebykey-default-eq-env

  stream.sync();
  REQUIRE(error == cudaSuccess);
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
  REQUIRE(num_selected == expected_num_selected);
}

CUB_TEST("cub::DeviceSelect::UniqueByKey accepts default env without equality_op", "[select][env]", CUB_SMALL)
{
  // Same expectations as other UniqueByKey tests, but call the 6-arg overload.
  auto keys_in      = thrust::device_vector<int>{0, 2, 2, 9, 5, 5, 5, 8};
  auto values_in    = thrust::device_vector<int>{1, 2, 3, 4, 5, 6, 7, 8};
  auto keys_out     = thrust::device_vector<int>(5);
  auto values_out   = thrust::device_vector<int>(5);
  auto num_selected = thrust::device_vector<int>(1);

  auto error = cub::DeviceSelect::UniqueByKey(
    keys_in.begin(), values_in.begin(), keys_out.begin(), values_out.begin(), num_selected.begin(), keys_in.size());

  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSelect::UniqueByKey with default env failed with status: " << error << '\n';
  }

  const thrust::device_vector<int> expected_keys{0, 2, 9, 5, 8};
  const thrust::device_vector<int> expected_values{1, 2, 4, 5, 8};
  const thrust::device_vector<int> expected_num_selected{5};

  REQUIRE(error == cudaSuccess);
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
  REQUIRE(num_selected == expected_num_selected);
}

#if _CCCL_STD_VER >= 2020

// example-begin unique-by-key-policy-selector
struct UniqueByKeyPolicySelector
{
  __host__ __device__ constexpr auto operator()(cuda::compute_capability cc) const -> cub::UniqueByKeyPolicy
  {
    return {.threads_per_block = 256,
            .items_per_thread  = cc > cuda::compute_capability{9, 0} ? 12 : 10,
            .load_algorithm    = cub::BLOCK_LOAD_DIRECT,
            .load_modifier     = cub::LOAD_DEFAULT,
            .scan_algorithm    = cub::BLOCK_SCAN_WARP_SCANS,
            .lookback_delay    = cub::LookbackDelayPolicy{
              .kind = cub::LookbackDelayAlgorithm::fixed_delay, .delay = 350, .l2_write_latency = 450}};
  }
};
// example-end unique-by-key-policy-selector

CUB_TEST("cub::DeviceSelect::UniqueByKey accepts a custom policy selector", "[select_unique_by_key][env]", CUB_SMALL)
{
  // example-begin unique-by-key-tuning
  auto keys_in          = thrust::device_vector<int>{0, 2, 2, 9, 5, 5, 5, 8};
  auto values_in        = thrust::device_vector<int>{0, 1, 2, 3, 4, 5, 6, 7};
  auto keys_out         = thrust::device_vector<int>(8, thrust::no_init);
  auto values_out       = thrust::device_vector<int>(8, thrust::no_init);
  auto num_selected_out = thrust::device_vector<int>(1, thrust::no_init);

  const auto error = cub::DeviceSelect::UniqueByKey(
    keys_in.begin(),
    values_in.begin(),
    keys_out.begin(),
    values_out.begin(),
    num_selected_out.begin(),
    keys_in.size(),
    cuda::execution::tune(UniqueByKeyPolicySelector{}));
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSelect::UniqueByKey failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected_keys{0, 2, 9, 5, 8};
  thrust::device_vector<int> expected_values{0, 1, 3, 4, 7};
  // example-end unique-by-key-tuning

  REQUIRE(error == cudaSuccess);
  const int n = num_selected_out[0];
  keys_out.resize(n);
  values_out.resize(n);
  CHECK(keys_out == expected_keys);
  CHECK(values_out == expected_values);
}

#endif // _CCCL_STD_VER >= 2020
