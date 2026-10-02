// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_reduce.cuh>

#include <thrust/device_vector.h>

#include <cuda/devices>
#include <cuda/stream>

#include <cstdint>
#include <iostream>

#include "cub_test_macros.h"

// example-begin reduce-non-commutative-op
// A number together with the power of ten that its digits span, e.g. {12, 100} for the digits "12"
struct digits_t
{
  std::uint64_t value;
  std::uint64_t scale;
};

// Appends the digits of `rhs` to those of `lhs`. Swapping the operands gives a different number.
struct append_digits_t
{
  __host__ __device__ digits_t operator()(digits_t lhs, digits_t rhs) const
  {
    return {lhs.value * rhs.scale + rhs.value, lhs.scale * rhs.scale};
  }
};
// example-end reduce-non-commutative-op

CUB_TEST("cub::DeviceReduce::ReduceNonCommutative API example", "[reduce][device]", CUB_SMALL)
{
  // example-begin reduce-non-commutative-two-phase
  auto digits          = thrust::device_vector<digits_t>{{1, 10}, {2, 10}, {3, 10}, {4, 10}, {5, 10}};
  auto result          = thrust::device_vector<digits_t>(1, thrust::no_init);
  const digits_t empty = {0, 1};

  // Determine temporary device storage requirements
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::ReduceNonCommutative(
    nullptr, temp_storage_bytes, digits.begin(), result.begin(), digits.size(), append_digits_t{}, empty);

  // Allocate temporary storage
  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes, thrust::no_init);

  // Run the reduction
  cub::DeviceReduce::ReduceNonCommutative(
    thrust::raw_pointer_cast(temp_storage.data()),
    temp_storage_bytes,
    digits.begin(),
    result.begin(),
    digits.size(),
    append_digits_t{},
    empty);

  // result <-- [{12345, 100000}]
  // example-end reduce-non-commutative-two-phase

  const digits_t parsed = result[0];
  REQUIRE(parsed.value == 12345);
  REQUIRE(parsed.scale == 100000);
}

CUB_TEST("cub::DeviceReduce::ReduceNonCommutative env API example", "[reduce][device]", CUB_SMALL)
{
  // example-begin reduce-non-commutative-env
  auto digits          = thrust::device_vector<digits_t>{{1, 10}, {2, 10}, {3, 10}, {4, 10}, {5, 10}};
  auto result          = thrust::device_vector<digits_t>(1, thrust::no_init);
  const digits_t empty = {0, 1};

  const cuda::stream stream{cuda::devices[0]};

  auto error = cub::DeviceReduce::ReduceNonCommutative(
    digits.begin(), result.begin(), digits.size(), append_digits_t{}, empty, stream);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::ReduceNonCommutative failed with status: " << error << '\n';
  }

  // result <-- [{12345, 100000}]
  // example-end reduce-non-commutative-env

  stream.sync();
  REQUIRE(error == cudaSuccess);
  const digits_t parsed = result[0];
  REQUIRE(parsed.value == 12345);
  REQUIRE(parsed.scale == 100000);
}
