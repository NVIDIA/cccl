// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_set_operations.cuh>

#include <thrust/device_vector.h>

#include <cuda/devices>
#include <cuda/std/functional>
#include <cuda/stream>

#include <iostream>

#include "cub_test_macros.h"

// All examples below operate on the same two sorted key sequences, so that the different key-value set operations can
// be compared side by side purely by their results:
//   keys1 = {0, 2, 4, 5, 7}
//   keys2 = {1, 2, 3, 5}
// Every key carries a value that encodes which input it came from (first-input value = key * 10, second-input value =
// key * 10 + 1), so the result values show where each emitted key was gathered from.

CUB_TEST("cub::detail::DeviceSetOps::SetDifferencePairs accepts an environment", "[set_ops][env]", CUB_SMALL)
{
  // example-begin set-difference-pairs-env
  auto keys1        = thrust::device_vector<int>{0, 2, 4, 5, 7};
  auto values1      = thrust::device_vector<int>{0, 20, 40, 50, 70};
  auto keys2        = thrust::device_vector<int>{1, 2, 3, 5};
  auto values2      = thrust::device_vector<int>{11, 21, 31, 51};
  auto keys_out     = thrust::device_vector<int>(keys1.size() + keys2.size(), thrust::no_init);
  auto values_out   = thrust::device_vector<int>(keys1.size() + keys2.size(), thrust::no_init);
  auto num_selected = thrust::device_vector<int>(1, thrust::no_init);

  const cuda::stream stream{cuda::devices[0]};

  auto error = cub::detail::DeviceSetOps::SetDifferencePairs(
    keys1.begin(),
    values1.begin(),
    keys1.size(),
    keys2.begin(),
    values2.begin(),
    keys2.size(),
    keys_out.begin(),
    values_out.begin(),
    num_selected.begin(),
    cuda::std::less<>{},
    cuda::stream_ref{stream});
  if (error != cudaSuccess)
  {
    std::cerr << "cub::detail::DeviceSetOps::SetDifferencePairs failed with status: " << error << '\n';
  }

  // keys present in keys1 but not in keys2, each with its first-input value
  const thrust::device_vector<int> expected_keys{0, 4, 7};
  const thrust::device_vector<int> expected_values{0, 40, 70};
  // example-end set-difference-pairs-env

  stream.sync();
  REQUIRE(error == cudaSuccess);
  keys_out.resize(num_selected[0]);
  values_out.resize(num_selected[0]);
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

CUB_TEST("cub::detail::DeviceSetOps::SetUnionPairs accepts an environment", "[set_ops][env]", CUB_SMALL)
{
  // example-begin set-union-pairs-env
  auto keys1        = thrust::device_vector<int>{0, 2, 4, 5, 7};
  auto values1      = thrust::device_vector<int>{0, 20, 40, 50, 70};
  auto keys2        = thrust::device_vector<int>{1, 2, 3, 5};
  auto values2      = thrust::device_vector<int>{11, 21, 31, 51};
  auto keys_out     = thrust::device_vector<int>(keys1.size() + keys2.size(), thrust::no_init);
  auto values_out   = thrust::device_vector<int>(keys1.size() + keys2.size(), thrust::no_init);
  auto num_selected = thrust::device_vector<int>(1, thrust::no_init);

  const cuda::stream stream{cuda::devices[0]};

  auto error = cub::detail::DeviceSetOps::SetUnionPairs(
    keys1.begin(),
    values1.begin(),
    keys1.size(),
    keys2.begin(),
    values2.begin(),
    keys2.size(),
    keys_out.begin(),
    values_out.begin(),
    num_selected.begin(),
    cuda::std::less<>{},
    cuda::stream_ref{stream});
  if (error != cudaSuccess)
  {
    std::cerr << "cub::detail::DeviceSetOps::SetUnionPairs failed with status: " << error << '\n';
  }

  // keys present in either input; for keys in both, the first input's value is kept
  const thrust::device_vector<int> expected_keys{0, 1, 2, 3, 4, 5, 7};
  const thrust::device_vector<int> expected_values{0, 11, 20, 31, 40, 50, 70};
  // example-end set-union-pairs-env

  stream.sync();
  REQUIRE(error == cudaSuccess);
  keys_out.resize(num_selected[0]);
  values_out.resize(num_selected[0]);
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

CUB_TEST("cub::detail::DeviceSetOps::SetIntersectionPairs accepts an environment", "[set_ops][env]", CUB_SMALL)
{
  // example-begin set-intersection-pairs-env
  auto keys1        = thrust::device_vector<int>{0, 2, 4, 5, 7};
  auto values1      = thrust::device_vector<int>{0, 20, 40, 50, 70};
  auto keys2        = thrust::device_vector<int>{1, 2, 3, 5};
  auto values2      = thrust::device_vector<int>{11, 21, 31, 51};
  auto keys_out     = thrust::device_vector<int>(keys1.size() + keys2.size(), thrust::no_init);
  auto values_out   = thrust::device_vector<int>(keys1.size() + keys2.size(), thrust::no_init);
  auto num_selected = thrust::device_vector<int>(1, thrust::no_init);

  const cuda::stream stream{cuda::devices[0]};

  auto error = cub::detail::DeviceSetOps::SetIntersectionPairs(
    keys1.begin(),
    values1.begin(),
    keys1.size(),
    keys2.begin(),
    values2.begin(),
    keys2.size(),
    keys_out.begin(),
    values_out.begin(),
    num_selected.begin(),
    cuda::std::less<>{},
    cuda::stream_ref{stream});
  if (error != cudaSuccess)
  {
    std::cerr << "cub::detail::DeviceSetOps::SetIntersectionPairs failed with status: " << error << '\n';
  }

  // keys present in both inputs, each with its first-input value
  const thrust::device_vector<int> expected_keys{2, 5};
  const thrust::device_vector<int> expected_values{20, 50};
  // example-end set-intersection-pairs-env

  stream.sync();
  REQUIRE(error == cudaSuccess);
  keys_out.resize(num_selected[0]);
  values_out.resize(num_selected[0]);
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

CUB_TEST("cub::detail::DeviceSetOps::SetSymmetricDifferencePairs accepts an environment", "[set_ops][env]", CUB_SMALL)
{
  // example-begin set-symmetric-difference-pairs-env
  auto keys1        = thrust::device_vector<int>{0, 2, 4, 5, 7};
  auto values1      = thrust::device_vector<int>{0, 20, 40, 50, 70};
  auto keys2        = thrust::device_vector<int>{1, 2, 3, 5};
  auto values2      = thrust::device_vector<int>{11, 21, 31, 51};
  auto keys_out     = thrust::device_vector<int>(keys1.size() + keys2.size(), thrust::no_init);
  auto values_out   = thrust::device_vector<int>(keys1.size() + keys2.size(), thrust::no_init);
  auto num_selected = thrust::device_vector<int>(1, thrust::no_init);

  const cuda::stream stream{cuda::devices[0]};

  auto error = cub::detail::DeviceSetOps::SetSymmetricDifferencePairs(
    keys1.begin(),
    values1.begin(),
    keys1.size(),
    keys2.begin(),
    values2.begin(),
    keys2.size(),
    keys_out.begin(),
    values_out.begin(),
    num_selected.begin(),
    cuda::std::less<>{},
    cuda::stream_ref{stream});
  if (error != cudaSuccess)
  {
    std::cerr << "cub::detail::DeviceSetOps::SetSymmetricDifferencePairs failed with status: " << error << '\n';
  }

  // keys present in exactly one input, each with the value from the input it came from
  const thrust::device_vector<int> expected_keys{0, 1, 3, 4, 7};
  const thrust::device_vector<int> expected_values{0, 11, 31, 40, 70};
  // example-end set-symmetric-difference-pairs-env

  stream.sync();
  REQUIRE(error == cudaSuccess);
  keys_out.resize(num_selected[0]);
  values_out.resize(num_selected[0]);
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}
