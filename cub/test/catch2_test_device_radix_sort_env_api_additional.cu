// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"
// above header needs to be included first

#include <cub/device/device_radix_sort.cuh>

#include <thrust/device_vector.h>

#include <cuda/devices>
#include <cuda/stream>

#include <iostream>

#include "cub_test_macros.h"

CUB_TEST("Device radix sort pairs example uses environment", "[radix_sort][env]", CUB_SMALL)
{
  // example-begin radix-sort-pairs-env
  auto keys_in    = thrust::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out   = thrust::device_vector<int>(7);
  auto values_in  = thrust::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out = thrust::device_vector<int>(7);

  const cuda::stream stream{cuda::devices[0]};
  const cuda::stream_ref env{stream};

  auto error = cub::DeviceRadixSort::SortPairs(
    keys_in.data().get(),
    keys_out.data().get(),
    values_in.data().get(),
    values_out.data().get(),
    static_cast<int>(keys_in.size()),
    0,
    sizeof(int) * 8,
    env);

  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceRadixSort::SortPairs failed with status: " << error << '\n';
  }

  stream.sync();

  const thrust::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  const thrust::device_vector<int> expected_values{5, 4, 3, 1, 2, 0, 6};
  // example-end radix-sort-pairs-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

CUB_TEST("Device radix sort pairs descending example uses environment", "[radix_sort][env]", CUB_SMALL)
{
  // example-begin radix-sort-pairs-descending-env
  auto keys_in    = thrust::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out   = thrust::device_vector<int>(7);
  auto values_in  = thrust::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out = thrust::device_vector<int>(7);

  const cuda::stream stream{cuda::devices[0]};
  const cuda::stream_ref env{stream};

  auto error = cub::DeviceRadixSort::SortPairsDescending(
    keys_in.data().get(),
    keys_out.data().get(),
    values_in.data().get(),
    values_out.data().get(),
    static_cast<int>(keys_in.size()),
    0,
    sizeof(int) * 8,
    env);

  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceRadixSort::SortPairsDescending failed with status: " << error << '\n';
  }

  stream.sync();

  const thrust::device_vector<int> expected_keys{9, 8, 7, 6, 5, 3, 0};
  const thrust::device_vector<int> expected_values{6, 0, 2, 1, 3, 4, 5};
  // example-end radix-sort-pairs-descending-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

CUB_TEST("Device radix sort keys example uses environment", "[radix_sort][env]", CUB_SMALL)
{
  // example-begin radix-sort-keys-env
  auto keys_in  = thrust::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out = thrust::device_vector<int>(7);

  const cuda::stream stream{cuda::devices[0]};
  const cuda::stream_ref env{stream};

  auto error = cub::DeviceRadixSort::SortKeys(
    keys_in.data().get(), keys_out.data().get(), static_cast<int>(keys_in.size()), 0, sizeof(int) * 8, env);

  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceRadixSort::SortKeys failed with status: " << error << '\n';
  }

  stream.sync();

  const thrust::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  // example-end radix-sort-keys-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(keys_out == expected_keys);
}

CUB_TEST("Device radix sort keys DB example uses environment", "[radix_sort][env]", CUB_SMALL)
{
  // example-begin radix-sort-keys-db-env
  thrust::device_vector<int> keys_buf0{8, 6, 7, 5, 3, 0, 9};
  thrust::device_vector<int> keys_buf1(7);

  cub::DoubleBuffer<int> d_keys(keys_buf0.data().get(), keys_buf1.data().get());

  const cuda::stream stream{cuda::devices[0]};
  const cuda::stream_ref env{stream};

  auto error = cub::DeviceRadixSort::SortKeys(d_keys, static_cast<int>(keys_buf0.size()), 0, sizeof(int) * 8, env);

  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceRadixSort::SortKeys (DoubleBuffer) failed with status: " << error << '\n';
  }

  stream.sync();

  const thrust::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  // example-end radix-sort-keys-db-env

  REQUIRE(error == cudaSuccess);
  auto& keys = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  REQUIRE(keys == expected_keys);
}

CUB_TEST("Device radix sort keys descending example uses environment", "[radix_sort][env]", CUB_SMALL)
{
  // example-begin radix-sort-keys-descending-env
  auto keys_in  = thrust::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out = thrust::device_vector<int>(7);

  const cuda::stream stream{cuda::devices[0]};
  const cuda::stream_ref env{stream};

  auto error = cub::DeviceRadixSort::SortKeysDescending(
    keys_in.data().get(), keys_out.data().get(), static_cast<int>(keys_in.size()), 0, sizeof(int) * 8, env);

  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceRadixSort::SortKeysDescending failed with status: " << error << '\n';
  }

  stream.sync();

  const thrust::device_vector<int> expected_keys{9, 8, 7, 6, 5, 3, 0};
  // example-end radix-sort-keys-descending-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(keys_out == expected_keys);
}

CUB_TEST("Device radix sort keys descending DB example uses environment", "[radix_sort][env]", CUB_SMALL)
{
  // example-begin radix-sort-keys-descending-db-env
  thrust::device_vector<int> keys_buf0{8, 6, 7, 5, 3, 0, 9};
  thrust::device_vector<int> keys_buf1(7);

  cub::DoubleBuffer<int> d_keys(keys_buf0.data().get(), keys_buf1.data().get());

  const cuda::stream stream{cuda::devices[0]};
  const cuda::stream_ref env{stream};

  auto error =
    cub::DeviceRadixSort::SortKeysDescending(d_keys, static_cast<int>(keys_buf0.size()), 0, sizeof(int) * 8, env);

  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceRadixSort::SortKeysDescending (DoubleBuffer) failed with status: " << error << '\n';
  }

  stream.sync();

  const thrust::device_vector<int> expected_keys{9, 8, 7, 6, 5, 3, 0};
  // example-end radix-sort-keys-descending-db-env

  REQUIRE(error == cudaSuccess);
  auto& keys = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  REQUIRE(keys == expected_keys);
}
