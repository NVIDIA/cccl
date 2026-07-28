// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_transform.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuda/iterator>

#include <c2h/catch2_test_helper.h>

// need a separate function because the ext. lambda needs to be enclosed by a function with external linkage on Windows
void test_transform_many_many_api()
{
  // example-begin transform-many-many
  auto input1 = thrust::device_vector<int>{0, -1, 2, -3, 4, -5};
  auto input2 = thrust::device_vector<double>{5.2, 3.1, -1.1, 3.0, 3.2, 0.0};
  auto op     = [] __device__(int a, double b) -> cuda::std::tuple<double, bool> {
    const double product = a * b;
    return {product, product < 0};
  };

  auto result1 = thrust::device_vector<double>(input1.size(), thrust::no_init);
  auto result2 = thrust::device_vector<bool>(input1.size(), thrust::no_init);
  cub::DeviceTransform::Transform(
    cuda::std::tuple{input1.begin(), input2.begin()},
    cuda::std::tuple{result1.begin(), result2.begin()},
    input1.size(),
    op);

  const auto expected1 = thrust::host_vector<double>{0, -3.1, -2.2, -9, 12.8, -0};
  const auto expected2 = thrust::host_vector<bool>{false, true, true, true, false, false};
  // example-end transform-many-many
  CHECK(result1 == expected1);
  CHECK(result2 == expected2);
}

C2H_TEST("DeviceTransform::Transform many->many API example", "[device][device_transform]")
{
  test_transform_many_many_api();
}

// need a separate function because the ext. lambda needs to be enclosed by a function with external linkage on Windows
void test_transform_api()
{
  // example-begin transform-many
  auto input1 = thrust::device_vector<int>{0, -2, 5, 3};
  auto input2 = thrust::device_vector<float>{5.2f, 3.1f, -1.1f, 3.0f};
  auto input3 = cuda::counting_iterator<int>{100};
  auto op     = [] __device__(int a, float b, int c) {
    return (static_cast<float>(a) + b) * static_cast<float>(c);
  };

  auto result = thrust::device_vector<int>(input1.size(), thrust::no_init);
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

void test_transform_if_single_api()
{
  // example-begin transform-if-single
  auto input     = thrust::device_vector<int>{0, -1, 2, -3, 4, -5};
  auto predicate = [] __device__(int value) {
    return value < 0;
  };
  auto op = [] __device__(int value) {
    return value * 2;
  };

  auto result = thrust::device_vector<int>(input.size()); // initialized to zeros
  cub::DeviceTransform::TransformIf(input.begin(), result.begin(), input.size(), predicate, op);

  const auto expected = thrust::host_vector<float>{0, -2, 0, -6, 0, -10};
  // example-end transform-if-single
  CHECK(result == expected);
}

C2H_TEST("DeviceTransform::TransformIf single-input API example", "[device][device_transform]")
{
  test_transform_if_single_api();
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

  auto result = thrust::device_vector<int>(input1.size(), thrust::no_init);
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

// Guard tests: each public DeviceTransform method must resolve unambiguously
// to the legacy temp-storage overload when called in its minimal form
// (no explicit stream, all defaults left implicit), even though the env and
// bare-stream overloads are also in scope. If the env-overload SFINAE drifts,
// these become "ambiguous overload" compile errors.

struct transform_noop_t
{
  __device__ int operator()(int x) const
  {
    return x;
  }
};

struct generate_zero_t
{
  __device__ int operator()() const
  {
    return 0;
  }
};

struct always_true_pred_t
{
  __device__ bool operator()(int) const
  {
    return true;
  }
};

C2H_TEST("DeviceTransform::Transform legacy size-query is unambiguous", "[device_transform][device]")
{
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  int* d_in                 = nullptr;
  int* d_out                = nullptr;
  int n                     = 0;

  REQUIRE(cudaSuccess
          == cub::DeviceTransform::Transform(d_temp_storage, temp_storage_bytes, d_in, d_out, n, transform_noop_t{}));
}

C2H_TEST("DeviceTransform::Generate legacy size-query is unambiguous", "[device_transform][device]")
{
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  int* d_out                = nullptr;
  int n                     = 0;

  REQUIRE(
    cudaSuccess == cub::DeviceTransform::Generate(d_temp_storage, temp_storage_bytes, d_out, n, generate_zero_t{}));
}

C2H_TEST("DeviceTransform::Fill legacy size-query is unambiguous", "[device_transform][device]")
{
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  int* d_out                = nullptr;
  int n                     = 0;
  int value                 = 0;

  REQUIRE(cudaSuccess == cub::DeviceTransform::Fill(d_temp_storage, temp_storage_bytes, d_out, n, value));
}

C2H_TEST("DeviceTransform::TransformIf legacy size-query is unambiguous", "[device_transform][device]")
{
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  int* d_in                 = nullptr;
  int* d_out                = nullptr;
  int n                     = 0;

  REQUIRE(cudaSuccess
          == cub::DeviceTransform::TransformIf(
            d_temp_storage, temp_storage_bytes, d_in, d_out, n, always_true_pred_t{}, transform_noop_t{}));
}

C2H_TEST("DeviceTransform::TransformStableArgumentAddresses legacy size-query is unambiguous",
         "[device_transform][device]")
{
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  int* d_in                 = nullptr;
  int* d_out                = nullptr;
  int n                     = 0;

  REQUIRE(cudaSuccess
          == cub::DeviceTransform::TransformStableArgumentAddresses(
            d_temp_storage, temp_storage_bytes, d_in, d_out, n, transform_noop_t{}));
}

// todo(giannis): guards for tuple-input variants (Transform many->many, TransformIf many->one,
// TransformStableArgumentAddresses many->one) — these need cuda::std::tuple setup; defer.
