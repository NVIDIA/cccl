// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

// example-begin adjacent-difference-includes
#include <cub/device/device_adjacent_difference.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cstddef>
#include <iostream>
// example-end adjacent-difference-includes

// example-begin custom-difference
struct custom_difference
{
  template <typename DataType>
  __device__ DataType operator()(const DataType& lhs, const DataType& rhs)
  {
    return lhs - rhs;
  }
};
// example-end custom-difference

#include "cub_test_macros.h"

// Guard: the legacy memory-size query call with all defaults (no explicit difference_op,
// no explicit stream) must resolve unambiguously to the legacy temp-storage overload
// when the env passthrough overload is also visible. If the env overload's SFINAE
// is too loose, this becomes "ambiguous overload" or silently dispatches to env.

CUB_TEST("DeviceAdjacentDifference::SubtractLeftCopy legacy size-query is unambiguous",
         "[adjacent_difference][device]",
         CUB_SMALL)
{
  int* d_in    = nullptr;
  int* d_out   = nullptr;
  size_t bytes = 0;
  const int n  = 0;

  REQUIRE(cudaSuccess == cub::DeviceAdjacentDifference::SubtractLeftCopy(nullptr, bytes, d_in, d_out, n));
}

CUB_TEST("DeviceAdjacentDifference::SubtractLeft legacy size-query is unambiguous",
         "[adjacent_difference][device]",
         CUB_SMALL)
{
  int* d_in    = nullptr;
  size_t bytes = 0;
  const int n  = 0;

  REQUIRE(cudaSuccess == cub::DeviceAdjacentDifference::SubtractLeft(nullptr, bytes, d_in, n));
}

CUB_TEST("DeviceAdjacentDifference::SubtractRightCopy legacy size-query is unambiguous",
         "[adjacent_difference][device]",
         CUB_SMALL)
{
  int* d_in    = nullptr;
  int* d_out   = nullptr;
  size_t bytes = 0;
  const int n  = 0;

  REQUIRE(cudaSuccess == cub::DeviceAdjacentDifference::SubtractRightCopy(nullptr, bytes, d_in, d_out, n));
}

CUB_TEST("DeviceAdjacentDifference::SubtractRight legacy size-query is unambiguous",
         "[adjacent_difference][device]",
         CUB_SMALL)
{
  int* d_in    = nullptr;
  size_t bytes = 0;
  const int n  = 0;

  REQUIRE(cudaSuccess == cub::DeviceAdjacentDifference::SubtractRight(nullptr, bytes, d_in, n));
}

CUB_TEST("DeviceAdjacentDifference::SubtractLeftCopy API example", "[adjacent_difference][device]", CUB_SMALL)
{
  // example-begin subtract-left-copy
  auto input  = thrust::device_vector<int>{1, 2, 1, 2, 1, 2, 1, 2};
  auto output = thrust::device_vector<int>(input.size(), thrust::no_init);

  // Determine temporary device storage requirements.
  std::size_t temp_storage_bytes = 0;
  const auto query_error         = cub::DeviceAdjacentDifference::SubtractLeftCopy(
    nullptr, temp_storage_bytes, input.begin(), output.begin(), input.size(), custom_difference{});
  if (query_error != cudaSuccess)
  {
    std::cerr << "cub::DeviceAdjacentDifference::SubtractLeftCopy storage query failed: " << query_error << '\n';
  }

  // Allocate temporary storage and run the operation.
  thrust::device_vector<char> temp_storage(temp_storage_bytes, thrust::no_init);
  const auto error = cub::DeviceAdjacentDifference::SubtractLeftCopy(
    temp_storage.data().get(), temp_storage_bytes, input.begin(), output.begin(), input.size(), custom_difference{});
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceAdjacentDifference::SubtractLeftCopy failed: " << error << '\n';
  }

  const thrust::host_vector<int> expected{1, 1, -1, 1, -1, 1, -1, 1};
  // example-end subtract-left-copy
  REQUIRE(query_error == cudaSuccess);
  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
  REQUIRE(input == thrust::host_vector<int>{1, 2, 1, 2, 1, 2, 1, 2});
}

CUB_TEST("DeviceAdjacentDifference::SubtractLeft API example", "[adjacent_difference][device]", CUB_SMALL)
{
  // example-begin subtract-left-in-place
  auto data = thrust::device_vector<int>{1, 2, 1, 2, 1, 2, 1, 2};

  // Determine temporary device storage requirements.
  std::size_t temp_storage_bytes = 0;
  const auto query_error         = cub::DeviceAdjacentDifference::SubtractLeft(
    nullptr, temp_storage_bytes, data.begin(), data.size(), custom_difference{});
  if (query_error != cudaSuccess)
  {
    std::cerr << "cub::DeviceAdjacentDifference::SubtractLeft storage query failed: " << query_error << '\n';
  }

  // Allocate temporary storage and run the operation.
  thrust::device_vector<char> temp_storage(temp_storage_bytes, thrust::no_init);
  const auto error = cub::DeviceAdjacentDifference::SubtractLeft(
    temp_storage.data().get(), temp_storage_bytes, data.begin(), data.size(), custom_difference{});
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceAdjacentDifference::SubtractLeft failed: " << error << '\n';
  }

  const thrust::host_vector<int> expected{1, 1, -1, 1, -1, 1, -1, 1};
  // example-end subtract-left-in-place
  REQUIRE(query_error == cudaSuccess);
  REQUIRE(error == cudaSuccess);
  REQUIRE(data == expected);
}

CUB_TEST("DeviceAdjacentDifference::SubtractRightCopy API example", "[adjacent_difference][device]", CUB_SMALL)
{
  // example-begin subtract-right-copy
  auto input  = thrust::device_vector<int>{1, 2, 1, 2, 1, 2, 1, 2};
  auto output = thrust::device_vector<int>(input.size(), thrust::no_init);

  // Determine temporary device storage requirements.
  std::size_t temp_storage_bytes = 0;
  const auto query_error         = cub::DeviceAdjacentDifference::SubtractRightCopy(
    nullptr, temp_storage_bytes, input.begin(), output.begin(), input.size(), custom_difference{});
  if (query_error != cudaSuccess)
  {
    std::cerr << "cub::DeviceAdjacentDifference::SubtractRightCopy storage query failed: " << query_error << '\n';
  }

  // Allocate temporary storage and run the operation.
  thrust::device_vector<char> temp_storage(temp_storage_bytes, thrust::no_init);
  const auto error = cub::DeviceAdjacentDifference::SubtractRightCopy(
    temp_storage.data().get(), temp_storage_bytes, input.begin(), output.begin(), input.size(), custom_difference{});
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceAdjacentDifference::SubtractRightCopy failed: " << error << '\n';
  }

  const thrust::host_vector<int> expected{-1, 1, -1, 1, -1, 1, -1, 2};
  // example-end subtract-right-copy
  REQUIRE(query_error == cudaSuccess);
  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
  REQUIRE(input == thrust::host_vector<int>{1, 2, 1, 2, 1, 2, 1, 2});
}

CUB_TEST("DeviceAdjacentDifference::SubtractRight API example", "[adjacent_difference][device]", CUB_SMALL)
{
  // example-begin subtract-right-in-place
  auto data = thrust::device_vector<int>{1, 2, 1, 2, 1, 2, 1, 2};

  // Determine temporary device storage requirements.
  std::size_t temp_storage_bytes = 0;
  const auto query_error =
    cub::DeviceAdjacentDifference::SubtractRight(nullptr, temp_storage_bytes, data.begin(), data.size());
  if (query_error != cudaSuccess)
  {
    std::cerr << "cub::DeviceAdjacentDifference::SubtractRight storage query failed: " << query_error << '\n';
  }

  // Allocate temporary storage and run the operation.
  thrust::device_vector<char> temp_storage(temp_storage_bytes, thrust::no_init);
  const auto error = cub::DeviceAdjacentDifference::SubtractRight(
    temp_storage.data().get(), temp_storage_bytes, data.begin(), data.size());
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceAdjacentDifference::SubtractRight failed: " << error << '\n';
  }

  const thrust::host_vector<int> expected{-1, 1, -1, 1, -1, 1, -1, 2};
  // example-end subtract-right-in-place
  REQUIRE(query_error == cudaSuccess);
  REQUIRE(error == cudaSuccess);
  REQUIRE(data == expected);
}
