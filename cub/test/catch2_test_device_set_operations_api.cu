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

// All examples below operate on the same two sorted key sequences (and, for the *Pairs variants, the same associated
// values), so that the different set operations can be compared side by side purely by their results:
//   keys1   = {0, 2, 4, 5, 7}     values1 = {'a', 'b', 'c', 'd', 'e'}
//   keys2   = {1, 2, 3, 5}        values2 = {'A', 'B', 'C', 'D'}

CUB_TEST("cub::detail::DeviceSetOps::SetDifference accepts an environment", "[set_ops][env]", CUB_SMALL)
{
  // example-begin set-difference-env
  auto keys1        = thrust::device_vector<int>{0, 2, 4, 5, 7};
  auto keys2        = thrust::device_vector<int>{1, 2, 3, 5};
  auto result       = thrust::device_vector<int>(keys1.size() + keys2.size(), thrust::no_init);
  auto num_selected = thrust::device_vector<int>(1, thrust::no_init);

  const cuda::stream stream{cuda::devices[0]};

  auto error = cub::detail::DeviceSetOps::SetDifference(
    keys1.begin(),
    static_cast<int>(keys1.size()),
    keys2.begin(),
    static_cast<int>(keys2.size()),
    result.begin(),
    num_selected.begin(),
    cuda::std::less<>{},
    cuda::stream_ref{stream});
  if (error != cudaSuccess)
  {
    std::cerr << "cub::detail::DeviceSetOps::SetDifference failed with status: " << error << '\n';
  }
  stream.sync();
  result.resize(num_selected[0]);

  // keys present in keys1 but not in keys2
  const thrust::device_vector<int> expected{0, 4, 7};
  // example-end set-difference-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(result == expected);
}

CUB_TEST("cub::detail::DeviceSetOps::SetIntersection accepts an environment", "[set_ops][env]", CUB_SMALL)
{
  // example-begin set-intersection-env
  auto keys1        = thrust::device_vector<int>{0, 2, 4, 5, 7};
  auto keys2        = thrust::device_vector<int>{1, 2, 3, 5};
  auto result       = thrust::device_vector<int>(keys1.size() + keys2.size(), thrust::no_init);
  auto num_selected = thrust::device_vector<int>(1, thrust::no_init);

  const cuda::stream stream{cuda::devices[0]};

  auto error = cub::detail::DeviceSetOps::SetIntersection(
    keys1.begin(),
    static_cast<int>(keys1.size()),
    keys2.begin(),
    static_cast<int>(keys2.size()),
    result.begin(),
    num_selected.begin(),
    cuda::std::less<>{},
    cuda::stream_ref{stream});
  if (error != cudaSuccess)
  {
    std::cerr << "cub::detail::DeviceSetOps::SetIntersection failed with status: " << error << '\n';
  }
  stream.sync();
  result.resize(num_selected[0]);

  // keys present in both keys1 and keys2
  const thrust::device_vector<int> expected{2, 5};
  // example-end set-intersection-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(result == expected);
}

CUB_TEST("cub::detail::DeviceSetOps::SetSymmetricDifference accepts an environment", "[set_ops][env]", CUB_SMALL)
{
  // example-begin set-symmetric-difference-env
  auto keys1        = thrust::device_vector<int>{0, 2, 4, 5, 7};
  auto keys2        = thrust::device_vector<int>{1, 2, 3, 5};
  auto result       = thrust::device_vector<int>(keys1.size() + keys2.size(), thrust::no_init);
  auto num_selected = thrust::device_vector<int>(1, thrust::no_init);

  const cuda::stream stream{cuda::devices[0]};

  auto error = cub::detail::DeviceSetOps::SetSymmetricDifference(
    keys1.begin(),
    static_cast<int>(keys1.size()),
    keys2.begin(),
    static_cast<int>(keys2.size()),
    result.begin(),
    num_selected.begin(),
    cuda::std::less<>{},
    cuda::stream_ref{stream});
  if (error != cudaSuccess)
  {
    std::cerr << "cub::detail::DeviceSetOps::SetSymmetricDifference failed with status: " << error << '\n';
  }
  stream.sync();
  result.resize(num_selected[0]);

  // keys present in exactly one of keys1 and keys2
  const thrust::device_vector<int> expected{0, 1, 3, 4, 7};
  // example-end set-symmetric-difference-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(result == expected);
}

CUB_TEST("cub::detail::DeviceSetOps::SetUnion accepts an environment", "[set_ops][env]", CUB_SMALL)
{
  // example-begin set-union-env
  auto keys1        = thrust::device_vector<int>{0, 2, 4, 5, 7};
  auto keys2        = thrust::device_vector<int>{1, 2, 3, 5};
  auto result       = thrust::device_vector<int>(keys1.size() + keys2.size(), thrust::no_init);
  auto num_selected = thrust::device_vector<int>(1, thrust::no_init);

  const cuda::stream stream{cuda::devices[0]};

  auto error = cub::detail::DeviceSetOps::SetUnion(
    keys1.begin(),
    static_cast<int>(keys1.size()),
    keys2.begin(),
    static_cast<int>(keys2.size()),
    result.begin(),
    num_selected.begin(),
    cuda::std::less<>{},
    cuda::stream_ref{stream});
  if (error != cudaSuccess)
  {
    std::cerr << "cub::detail::DeviceSetOps::SetUnion failed with status: " << error << '\n';
  }
  stream.sync();
  result.resize(num_selected[0]);

  // keys present in either keys1 or keys2
  const thrust::device_vector<int> expected{0, 1, 2, 3, 4, 5, 7};
  // example-end set-union-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(result == expected);
}

CUB_TEST("cub::detail::DeviceSetOps::SetDifferencePairs accepts an environment", "[set_ops][env]", CUB_SMALL)
{
  // example-begin set-difference-pairs-env
  auto keys1   = thrust::device_vector<int>{0, 2, 4, 5, 7};
  auto values1 = thrust::device_vector<char>{'a', 'b', 'c', 'd', 'e'};
  auto keys2   = thrust::device_vector<int>{1, 2, 3, 5};
  auto values2 = thrust::device_vector<char>{'A', 'B', 'C', 'D'};

  auto result_keys   = thrust::device_vector<int>(keys1.size() + keys2.size(), thrust::no_init);
  auto result_values = thrust::device_vector<char>(keys1.size() + keys2.size(), thrust::no_init);
  auto num_selected  = thrust::device_vector<int>(1, thrust::no_init);

  const cuda::stream stream{cuda::devices[0]};

  auto error = cub::detail::DeviceSetOps::SetDifferencePairs(
    keys1.begin(),
    values1.begin(),
    static_cast<int>(keys1.size()),
    keys2.begin(),
    values2.begin(),
    static_cast<int>(keys2.size()),
    result_keys.begin(),
    result_values.begin(),
    num_selected.begin(),
    cuda::std::less<>{},
    cuda::stream_ref{stream});
  if (error != cudaSuccess)
  {
    std::cerr << "cub::detail::DeviceSetOps::SetDifferencePairs failed with status: " << error << '\n';
  }
  stream.sync();
  result_keys.resize(num_selected[0]);
  result_values.resize(num_selected[0]);

  // keys present in keys1 but not in keys2, each carrying its value from the first input
  const thrust::device_vector<int> expected_keys{0, 4, 7};
  const thrust::device_vector<char> expected_values{'a', 'c', 'e'};
  // example-end set-difference-pairs-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(result_keys == expected_keys);
  REQUIRE(result_values == expected_values);
}

CUB_TEST("cub::detail::DeviceSetOps::SetIntersectionPairs accepts an environment", "[set_ops][env]", CUB_SMALL)
{
  // example-begin set-intersection-pairs-env
  auto keys1   = thrust::device_vector<int>{0, 2, 4, 5, 7};
  auto values1 = thrust::device_vector<char>{'a', 'b', 'c', 'd', 'e'};
  auto keys2   = thrust::device_vector<int>{1, 2, 3, 5};
  auto values2 = thrust::device_vector<char>{'A', 'B', 'C', 'D'};

  auto result_keys   = thrust::device_vector<int>(keys1.size() + keys2.size(), thrust::no_init);
  auto result_values = thrust::device_vector<char>(keys1.size() + keys2.size(), thrust::no_init);
  auto num_selected  = thrust::device_vector<int>(1, thrust::no_init);

  const cuda::stream stream{cuda::devices[0]};

  auto error = cub::detail::DeviceSetOps::SetIntersectionPairs(
    keys1.begin(),
    values1.begin(),
    static_cast<int>(keys1.size()),
    keys2.begin(),
    values2.begin(),
    static_cast<int>(keys2.size()),
    result_keys.begin(),
    result_values.begin(),
    num_selected.begin(),
    cuda::std::less<>{},
    cuda::stream_ref{stream});
  if (error != cudaSuccess)
  {
    std::cerr << "cub::detail::DeviceSetOps::SetIntersectionPairs failed with status: " << error << '\n';
  }
  stream.sync();
  result_keys.resize(num_selected[0]);
  result_values.resize(num_selected[0]);

  // keys present in both inputs, each carrying its value from the first input
  const thrust::device_vector<int> expected_keys{2, 5};
  const thrust::device_vector<char> expected_values{'b', 'd'};
  // example-end set-intersection-pairs-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(result_keys == expected_keys);
  REQUIRE(result_values == expected_values);
}

CUB_TEST("cub::detail::DeviceSetOps::SetSymmetricDifferencePairs accepts an environment", "[set_ops][env]", CUB_SMALL)
{
  // example-begin set-symmetric-difference-pairs-env
  auto keys1   = thrust::device_vector<int>{0, 2, 4, 5, 7};
  auto values1 = thrust::device_vector<char>{'a', 'b', 'c', 'd', 'e'};
  auto keys2   = thrust::device_vector<int>{1, 2, 3, 5};
  auto values2 = thrust::device_vector<char>{'A', 'B', 'C', 'D'};

  auto result_keys   = thrust::device_vector<int>(keys1.size() + keys2.size(), thrust::no_init);
  auto result_values = thrust::device_vector<char>(keys1.size() + keys2.size(), thrust::no_init);
  auto num_selected  = thrust::device_vector<int>(1, thrust::no_init);

  const cuda::stream stream{cuda::devices[0]};

  auto error = cub::detail::DeviceSetOps::SetSymmetricDifferencePairs(
    keys1.begin(),
    values1.begin(),
    static_cast<int>(keys1.size()),
    keys2.begin(),
    values2.begin(),
    static_cast<int>(keys2.size()),
    result_keys.begin(),
    result_values.begin(),
    num_selected.begin(),
    cuda::std::less<>{},
    cuda::stream_ref{stream});
  if (error != cudaSuccess)
  {
    std::cerr << "cub::detail::DeviceSetOps::SetSymmetricDifferencePairs failed with status: " << error << '\n';
  }
  stream.sync();
  result_keys.resize(num_selected[0]);
  result_values.resize(num_selected[0]);

  // keys present in exactly one input; each carries its value from the input it came from
  const thrust::device_vector<int> expected_keys{0, 1, 3, 4, 7};
  const thrust::device_vector<char> expected_values{'a', 'A', 'C', 'c', 'e'};
  // example-end set-symmetric-difference-pairs-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(result_keys == expected_keys);
  REQUIRE(result_values == expected_values);
}

CUB_TEST("cub::detail::DeviceSetOps::SetUnionPairs accepts an environment", "[set_ops][env]", CUB_SMALL)
{
  // example-begin set-union-pairs-env
  auto keys1   = thrust::device_vector<int>{0, 2, 4, 5, 7};
  auto values1 = thrust::device_vector<char>{'a', 'b', 'c', 'd', 'e'};
  auto keys2   = thrust::device_vector<int>{1, 2, 3, 5};
  auto values2 = thrust::device_vector<char>{'A', 'B', 'C', 'D'};

  auto result_keys   = thrust::device_vector<int>(keys1.size() + keys2.size(), thrust::no_init);
  auto result_values = thrust::device_vector<char>(keys1.size() + keys2.size(), thrust::no_init);
  auto num_selected  = thrust::device_vector<int>(1, thrust::no_init);

  const cuda::stream stream{cuda::devices[0]};

  auto error = cub::detail::DeviceSetOps::SetUnionPairs(
    keys1.begin(),
    values1.begin(),
    static_cast<int>(keys1.size()),
    keys2.begin(),
    values2.begin(),
    static_cast<int>(keys2.size()),
    result_keys.begin(),
    result_values.begin(),
    num_selected.begin(),
    cuda::std::less<>{},
    cuda::stream_ref{stream});
  if (error != cudaSuccess)
  {
    std::cerr << "cub::detail::DeviceSetOps::SetUnionPairs failed with status: " << error << '\n';
  }
  stream.sync();
  result_keys.resize(num_selected[0]);
  result_values.resize(num_selected[0]);

  // keys present in either input; on a tie the key and value are taken from the first input
  const thrust::device_vector<int> expected_keys{0, 1, 2, 3, 4, 5, 7};
  const thrust::device_vector<char> expected_values{'a', 'A', 'b', 'C', 'c', 'd', 'e'};
  // example-end set-union-pairs-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(result_keys == expected_keys);
  REQUIRE(result_values == expected_values);
}
