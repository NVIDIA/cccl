// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_scan.cuh>

#include <thrust/device_vector.h>

#include <cuda/__device/arch_id.h>
#include <cuda/__iterator/constant_iterator.h>

#include "catch2_test_env_launch_helper.h"

DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::ExclusiveSumByKey, device_scan_exclusive_sum_by_key);
DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::ExclusiveScanByKey, device_scan_exclusive_scan_by_key);
DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::InclusiveSumByKey, device_scan_inclusive_sum_by_key);
DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::InclusiveScanByKey, device_scan_inclusive_scan_by_key);

// %PARAM% TEST_LAUNCH lid 0:1:2

#include <c2h/catch2_test_helper.h>

namespace stdexec = cuda::std::execution;

#if TEST_LAUNCH == 0
struct block_size_check_t
{
  int* ptr;

  __device__ int operator()(int a, int b)
  {
    *ptr = blockDim.x;
    return a + b;
  }
};

TEST_CASE("Device scan exclusive-sum-by-key works with default environment", "[scan][by_key][device]")
{
  auto num_items = 7;
  auto d_keys    = thrust::device_vector<int>{0, 0, 1, 1, 1, 2, 2};
  auto d_in      = thrust::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto d_out     = thrust::device_vector<int>(num_items);

  REQUIRE(cudaSuccess == cub::DeviceScan::ExclusiveSumByKey(d_keys.begin(), d_in.begin(), d_out.begin(), num_items));

  thrust::device_vector<int> expected{0, 8, 0, 7, 12, 0, 0};
  REQUIRE(d_out == expected);
}

TEST_CASE("Device scan exclusive-scan-by-key works with default environment", "[scan][by_key][device]")
{
  using num_items_t = int;
  using key_t       = int;
  using value_t     = int;
  using accum_t     = value_t;

  using selector_t = cub::detail::scan_by_key::policy_selector_from_types<key_t, accum_t, value_t, block_size_check_t>;

  int current_device{};
  REQUIRE(cudaSuccess == cudaGetDevice(&current_device));

  cudaDeviceProp device_props{};
  REQUIRE(cudaSuccess == cudaGetDeviceProperties(&device_props, current_device));

  const auto target_block_size =
    selector_t{}(cuda::to_arch_id(cuda::compute_capability{device_props.major, device_props.minor})).block_threads;

  num_items_t num_items = 1;
  auto d_keys           = thrust::device_vector<key_t>{0};
  c2h::device_vector<int> d_block_size(1);
  block_size_check_t block_size_check{thrust::raw_pointer_cast(d_block_size.data())};
  auto d_in  = cuda::constant_iterator(value_t{1});
  auto d_out = thrust::device_vector<value_t>(1);
  auto init  = value_t{0};

  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::ExclusiveScanByKey(d_keys.begin(), d_in, d_out.begin(), block_size_check, init, num_items));

  REQUIRE(d_out[0] == init);
  REQUIRE(d_block_size[0] == target_block_size);
}

TEST_CASE("Device scan inclusive-sum-by-key works with default environment", "[scan][by_key][device]")
{
  auto num_items = 7;
  auto d_keys    = thrust::device_vector<int>{0, 0, 1, 1, 1, 2, 2};
  auto d_in      = thrust::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto d_out     = thrust::device_vector<int>(num_items);

  REQUIRE(cudaSuccess == cub::DeviceScan::InclusiveSumByKey(d_keys.begin(), d_in.begin(), d_out.begin(), num_items));

  thrust::device_vector<int> expected{8, 14, 7, 12, 15, 0, 9};
  REQUIRE(d_out == expected);
}

TEST_CASE("Device scan inclusive-scan-by-key works with default environment", "[scan][by_key][device]")
{
  using num_items_t = int;
  using key_t       = int;
  using value_t     = int;
  using accum_t     = value_t;

  using selector_t = cub::detail::scan_by_key::policy_selector_from_types<key_t, accum_t, value_t, block_size_check_t>;

  int current_device{};
  REQUIRE(cudaSuccess == cudaGetDevice(&current_device));

  cudaDeviceProp device_props{};
  REQUIRE(cudaSuccess == cudaGetDeviceProperties(&device_props, current_device));

  const auto target_block_size =
    selector_t{}(cuda::to_arch_id(cuda::compute_capability{device_props.major, device_props.minor})).block_threads;

  num_items_t num_items = 1;
  auto d_keys           = thrust::device_vector<key_t>{0};
  c2h::device_vector<int> d_block_size(1);
  block_size_check_t block_size_check{thrust::raw_pointer_cast(d_block_size.data())};
  auto d_in  = cuda::constant_iterator(value_t{1});
  auto d_out = thrust::device_vector<value_t>(1);

  REQUIRE(cudaSuccess
          == cub::DeviceScan::InclusiveScanByKey(d_keys.begin(), d_in, d_out.begin(), block_size_check, num_items));

  REQUIRE(d_out[0] == value_t{1});
  REQUIRE(d_block_size[0] == target_block_size);
}

#endif

C2H_TEST("Device scan exclusive-sum-by-key uses environment", "[scan][by_key][device]")
{
  using num_items_t = int;

  num_items_t num_items = 7;
  auto d_keys           = thrust::device_vector<int>{0, 0, 1, 1, 1, 2, 2};
  auto d_in             = thrust::device_vector<float>{8.0f, 6.0f, 7.0f, 5.0f, 3.0f, 0.0f, 9.0f};
  auto d_out            = thrust::device_vector<float>(num_items);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::ExclusiveSumByKey(
      nullptr,
      expected_bytes_allocated,
      d_keys.begin(),
      d_in.begin(),
      d_out.begin(),
      num_items,
      cuda::std::equal_to<>{}));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_scan_exclusive_sum_by_key(d_keys.begin(), d_in.begin(), d_out.begin(), num_items, cuda::std::equal_to<>{}, env);

  thrust::device_vector<float> expected{0.0f, 8.0f, 0.0f, 7.0f, 12.0f, 0.0f, 0.0f};
  REQUIRE(d_out == expected);
}

C2H_TEST("Device scan exclusive-scan-by-key uses environment", "[scan][by_key][device]")
{
  using scan_op_t   = cuda::std::plus<>;
  using num_items_t = int;

  num_items_t num_items = 7;
  auto d_keys           = thrust::device_vector<int>{0, 0, 1, 1, 1, 2, 2};
  auto d_in             = thrust::device_vector<float>{8.0f, 6.0f, 7.0f, 5.0f, 3.0f, 0.0f, 9.0f};
  auto d_out            = thrust::device_vector<float>(num_items);
  auto init             = 0.0f;

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::ExclusiveScanByKey(
      nullptr,
      expected_bytes_allocated,
      d_keys.begin(),
      d_in.begin(),
      d_out.begin(),
      scan_op_t{},
      init,
      num_items,
      cuda::std::equal_to<>{}));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_scan_exclusive_scan_by_key(
    d_keys.begin(), d_in.begin(), d_out.begin(), scan_op_t{}, init, num_items, cuda::std::equal_to<>{}, env);

  thrust::device_vector<float> expected{0.0f, 8.0f, 0.0f, 7.0f, 12.0f, 0.0f, 0.0f};
  REQUIRE(d_out == expected);
}

C2H_TEST("Device scan inclusive-sum-by-key uses environment", "[scan][by_key][device]")
{
  using num_items_t = int;

  num_items_t num_items = 7;
  auto d_keys           = thrust::device_vector<int>{0, 0, 1, 1, 1, 2, 2};
  auto d_in             = thrust::device_vector<float>{8.0f, 6.0f, 7.0f, 5.0f, 3.0f, 0.0f, 9.0f};
  auto d_out            = thrust::device_vector<float>(num_items);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::InclusiveSumByKey(
      nullptr,
      expected_bytes_allocated,
      d_keys.begin(),
      d_in.begin(),
      d_out.begin(),
      num_items,
      cuda::std::equal_to<>{}));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_scan_inclusive_sum_by_key(d_keys.begin(), d_in.begin(), d_out.begin(), num_items, cuda::std::equal_to<>{}, env);

  thrust::device_vector<float> expected{8.0f, 14.0f, 7.0f, 12.0f, 15.0f, 0.0f, 9.0f};
  REQUIRE(d_out == expected);
}

C2H_TEST("Device scan inclusive-scan-by-key uses environment", "[scan][by_key][device]")
{
  using scan_op_t   = cuda::std::plus<>;
  using num_items_t = int;

  num_items_t num_items = 7;
  auto d_keys           = thrust::device_vector<int>{0, 0, 1, 1, 1, 2, 2};
  auto d_in             = thrust::device_vector<float>{8.0f, 6.0f, 7.0f, 5.0f, 3.0f, 0.0f, 9.0f};
  auto d_out            = thrust::device_vector<float>(num_items);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::InclusiveScanByKey(
      nullptr,
      expected_bytes_allocated,
      d_keys.begin(),
      d_in.begin(),
      d_out.begin(),
      scan_op_t{},
      num_items,
      cuda::std::equal_to<>{}));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_scan_inclusive_scan_by_key(
    d_keys.begin(), d_in.begin(), d_out.begin(), scan_op_t{}, num_items, cuda::std::equal_to<>{}, env);

  thrust::device_vector<float> expected{8.0f, 14.0f, 7.0f, 12.0f, 15.0f, 0.0f, 9.0f};
  REQUIRE(d_out == expected);
}
