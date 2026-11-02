// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_find.cuh>

#include <thrust/device_vector.h>

#include <iostream>

#include "cub_test_macros.h"

// example-begin find-if-predicate
struct is_greater_than_t
{
  int threshold;
  __host__ __device__ bool operator()(int value) const
  {
    return value > threshold;
  }
};
// example-end find-if-predicate

CUB_TEST("cub::DeviceFind::FindIf works with int data elements", "[find][device]", CUB_SMALL)
{
  // example-begin device-find-if
  constexpr int num_items         = 8;
  thrust::device_vector<int> d_in = {0, 1, 2, 3, 4, 5, 6, 7};
  thrust::device_vector<int> d_out(1, thrust::no_init);
  const is_greater_than_t predicate{4};

  size_t temp_storage_bytes = 0;
  const auto query_error =
    cub::DeviceFind::FindIf(nullptr, temp_storage_bytes, d_in.begin(), d_out.begin(), predicate, num_items);
  if (query_error != cudaSuccess)
  {
    std::cerr << "cub::DeviceFind::FindIf failed with status: " << query_error << '\n';
  }

  thrust::device_vector<char> temp_storage(temp_storage_bytes, thrust::no_init);

  const auto error = cub::DeviceFind::FindIf(
    thrust::raw_pointer_cast(temp_storage.data()),
    temp_storage_bytes,
    d_in.begin(),
    d_out.begin(),
    predicate,
    num_items);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceFind::FindIf failed with status: " << error << '\n';
  }

  const int expected = 5;
  // example-end device-find-if
  REQUIRE(query_error == cudaSuccess);
  REQUIRE(error == cudaSuccess);

  REQUIRE(d_out[0] == expected);
}

CUB_TEST("cub::DeviceFind::LowerBound works with int data elements", "[find][device]", CUB_SMALL)
{
  // example-begin device-lower-bound
  thrust::device_vector<int> d_range  = {0, 2, 4, 6, 8};
  thrust::device_vector<int> d_values = {1, 3, 5, 7};
  thrust::device_vector<int> d_output(4);

  size_t temp_storage_bytes = 0;
  const auto query_error    = cub::DeviceFind::LowerBound(
    nullptr,
    temp_storage_bytes,
    d_range.begin(),
    static_cast<int>(d_range.size()),
    d_values.begin(),
    static_cast<int>(d_values.size()),
    d_output.begin(),
    cuda::std::less{});
  if (query_error != cudaSuccess)
  {
    std::cerr << "cub::DeviceFind::LowerBound failed with status: " << query_error << '\n';
  }

  thrust::device_vector<char> temp_storage(temp_storage_bytes, thrust::no_init);

  const auto error = cub::DeviceFind::LowerBound(
    thrust::raw_pointer_cast(temp_storage.data()),
    temp_storage_bytes,
    d_range.begin(),
    static_cast<int>(d_range.size()),
    d_values.begin(),
    static_cast<int>(d_values.size()),
    d_output.begin(),
    cuda::std::less{});
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceFind::LowerBound failed with status: " << error << '\n';
  }

  const thrust::device_vector<int> expected = {1, 2, 3, 4};
  // example-end device-lower-bound
  REQUIRE(query_error == cudaSuccess);
  REQUIRE(error == cudaSuccess);

  REQUIRE(d_output == expected);
}

CUB_TEST("cub::DeviceFind::UpperBound works with int data elements", "[find][device]", CUB_SMALL)
{
  // example-begin device-upper-bound
  thrust::device_vector<int> d_range  = {0, 2, 4, 6, 8};
  thrust::device_vector<int> d_values = {1, 3, 5, 7};
  thrust::device_vector<int> d_output(4);

  size_t temp_storage_bytes = 0;
  const auto query_error    = cub::DeviceFind::UpperBound(
    nullptr,
    temp_storage_bytes,
    d_range.begin(),
    static_cast<int>(d_range.size()),
    d_values.begin(),
    static_cast<int>(d_values.size()),
    d_output.begin(),
    cuda::std::less{});
  if (query_error != cudaSuccess)
  {
    std::cerr << "cub::DeviceFind::UpperBound failed with status: " << query_error << '\n';
  }

  thrust::device_vector<char> temp_storage(temp_storage_bytes, thrust::no_init);

  const auto error = cub::DeviceFind::UpperBound(
    thrust::raw_pointer_cast(temp_storage.data()),
    temp_storage_bytes,
    d_range.begin(),
    static_cast<int>(d_range.size()),
    d_values.begin(),
    static_cast<int>(d_values.size()),
    d_output.begin(),
    cuda::std::less{});
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceFind::UpperBound failed with status: " << error << '\n';
  }

  const thrust::device_vector<int> expected = {1, 2, 3, 4};
  // example-end device-upper-bound
  REQUIRE(query_error == cudaSuccess);
  REQUIRE(error == cudaSuccess);

  REQUIRE(d_output == expected);
}

// Guard: the legacy memory-size query call with all defaults (no explicit stream)
// must resolve unambiguously to the legacy temp-storage overload when the env
// passthrough overload is also visible. If the env overload's SFINAE is too loose,
// this becomes "ambiguous overload" or silently dispatches to env.

CUB_TEST("DeviceFind::FindIf legacy size-query is unambiguous", "[find][device]", CUB_SMALL)
{
  int* d_in    = nullptr;
  int* d_out   = nullptr;
  size_t bytes = 0;
  const int n  = 0;

  REQUIRE(cudaSuccess == cub::DeviceFind::FindIf(nullptr, bytes, d_in, d_out, is_greater_than_t{0}, n));
}

CUB_TEST("DeviceFind::LowerBound legacy size-query is unambiguous", "[find][device]", CUB_SMALL)
{
  int* const d_range  = nullptr;
  int* const d_values = nullptr;
  int* const d_output = nullptr;
  size_t bytes        = 0;
  const int range_n   = 0;
  const int values_n  = 0;

  REQUIRE(
    cudaSuccess
    == cub::DeviceFind::LowerBound(nullptr, bytes, d_range, range_n, d_values, values_n, d_output, cuda::std::less{}));
}

CUB_TEST("DeviceFind::UpperBound legacy size-query is unambiguous", "[find][device]", CUB_SMALL)
{
  int* d_range       = nullptr;
  int* d_values      = nullptr;
  int* d_output      = nullptr;
  size_t bytes       = 0;
  const int range_n  = 0;
  const int values_n = 0;

  REQUIRE(
    cudaSuccess
    == cub::DeviceFind::UpperBound(nullptr, bytes, d_range, range_n, d_values, values_n, d_output, cuda::std::less{}));
}
