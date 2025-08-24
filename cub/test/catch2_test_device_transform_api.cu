// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_transform.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <c2h/catch2_test_helper.h>

// need a separate function because the ext. lambda needs to be enclosed by a function with external linkage on Windows
void test_transform_api()
{
  // example-begin transform-many
  auto input1 = thrust::device_vector<int>{0, -2, 5, 3};
  auto input2 = thrust::device_vector<float>{5.2f, 3.1f, -1.1f, 3.0f};
  auto input3 = thrust::counting_iterator<int>{100};
  auto op     = [] __device__(int a, float b, int c) {
    return (a + b) * c;
  };

  auto result = thrust::device_vector<int>(input1.size());
  cub::DeviceTransform::Transform(
    cuda::std::tuple{input1.begin(), input2.begin(), input3}, result.begin(), input1.size(), op);

  const auto expected = thrust::host_vector<float>{520, 111, 397, 618};
  // example-end transform-many
  CHECK(result == expected);
}

C2H_TEST("DeviceTransform::Transform API example", "[device][device_transform]")
{
  test_transform_api();
}

void test_transform_if_api()
{
  // example-begin transform-if
  auto input     = thrust::device_vector<int>{0, -1, 2, -3, 4, -5};
  auto predicate = [] __device__(int value) {
    return value < 0;
  };
  auto op = [] __device__(int value) {
    return value * 2;
  };

  auto result = thrust::device_vector<int>(input.size()); // initialized to zeros
  cub::DeviceTransform::TransformIf(cuda::std::tuple{input.begin()}, result.begin(), input.size(), predicate, op);

  const auto expected = thrust::host_vector<float>{0, -2, 0, -6, 0, -10};
  // example-end transform-if
  CHECK(result == expected);
}

C2H_TEST("DeviceTransform::TransformIf API example", "[device][device_transform]")
{
  test_transform_if_api();
}

// need a separate function because the ext. lambda needs to be enclosed by a function with external linkage on Windows
void test_transform_stable_api()
{
  // example-begin transform-many-stable
  auto input1 = thrust::device_vector<int>{0, -2, 5, 3};
  auto input2 = thrust::device_vector<int>{52, 31, -11, 30};

  auto* input1_ptr = thrust::raw_pointer_cast(input1.data());
  auto* input2_ptr = thrust::raw_pointer_cast(input2.data());

  auto op = [input1_ptr, input2_ptr] __device__(const int& a) -> int {
    const auto i = &a - input1_ptr; // we depend on the address of a
    return a + input2_ptr[i];
  };

  auto result = thrust::device_vector<int>(input1.size());
  cub::DeviceTransform::TransformStableArgumentAddresses(
    cuda::std::tuple{input1_ptr}, result.begin(), input1.size(), op);

  const auto expected = thrust::host_vector<float>{52, 29, -6, 33};
  // example-end transform-many-stable
  CHECK(result == expected);
}

C2H_TEST("DeviceTransform::TransformStableArgumentAddresses API example", "[device][device_transform]")
{
  test_transform_stable_api();
}
