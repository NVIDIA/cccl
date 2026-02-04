// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_partition.cuh>

#include <thrust/device_vector.h>

#include <cuda/iterator>

#include "catch2_test_device_select_common.cuh"
#include "catch2_test_env_launch_helper.h"

DECLARE_LAUNCH_WRAPPER(cub::DevicePartition::If, device_partition_if);
DECLARE_LAUNCH_WRAPPER(cub::DevicePartition::Flagged, device_partition_flagged);

// %PARAM% TEST_LAUNCH lid 0:1:2

#include <c2h/catch2_test_helper.h>

namespace stdexec = cuda::std::execution;

#if TEST_LAUNCH == 0

TEST_CASE("Device partition works with default environment", "[partition][device]")
{
  using value_t     = int;
  using num_items_t = int;

  num_items_t num_items = 8;
  auto d_in             = c2h::device_vector<value_t>{1, 2, 3, 4, 5, 6, 7, 8};
  auto d_out            = c2h::device_vector<value_t>(num_items);
  auto d_num_selected   = c2h::device_vector<int>(1);

  less_than_t<value_t> select_op{5};

  // launch wrapper always assumes the last argument is the environment
  REQUIRE(
    cudaSuccess == cub::DevicePartition::If(d_in.begin(), d_out.begin(), d_num_selected.begin(), num_items, select_op));

  c2h::device_vector<value_t> expected_output{1, 2, 3, 4, 8, 7, 6, 5};
  c2h::device_vector<int> expected_num_selected{4};

  REQUIRE(d_num_selected == expected_num_selected);
  REQUIRE(d_out == expected_output);
}

TEST_CASE("Device partition flagged works with default environment", "[partition][device]")
{
  using value_t     = int;
  using num_items_t = int;

  num_items_t num_items = 8;
  auto d_in             = c2h::device_vector<value_t>{1, 2, 3, 4, 5, 6, 7, 8};
  auto d_flags          = c2h::device_vector<char>{1, 0, 0, 1, 0, 1, 1, 0};
  auto d_out            = c2h::device_vector<value_t>(num_items);
  auto d_num_selected   = c2h::device_vector<int>(1);

  // launch wrapper always assumes the last argument is the environment
  REQUIRE(
    cudaSuccess
    == cub::DevicePartition::Flagged(d_in.begin(), d_flags.begin(), d_out.begin(), d_num_selected.begin(), num_items));

  c2h::device_vector<value_t> expected_output{1, 4, 6, 7, 8, 5, 3, 2};
  c2h::device_vector<int> expected_num_selected{4};

  REQUIRE(d_num_selected == expected_num_selected);
  REQUIRE(d_out == expected_output);
}

#endif

C2H_TEST("Device partition uses environment", "[partition][device]")
{
  using value_t     = int;
  using num_items_t = int;

  num_items_t num_items = 10;
  auto d_in             = c2h::device_vector<value_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto d_out            = c2h::device_vector<value_t>(num_items);
  auto d_num_selected   = c2h::device_vector<int>(1);

  less_than_t<value_t> select_op{6};

  size_t expected_bytes_allocated{};
  // calculate expected_bytes_allocated - call CUB API directly, not through wrapper
  REQUIRE(
    cudaSuccess
    == cub::DevicePartition::If(
      nullptr, expected_bytes_allocated, d_in.begin(), d_out.begin(), d_num_selected.begin(), num_items, select_op));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)}; // temp storage size

  device_partition_if(d_in.begin(), d_out.begin(), d_num_selected.begin(), num_items, select_op, env);

  // Items < 6: {1, 2, 3, 4, 5} at front, items >= 6: {6, 7, 8, 9, 10} at back in reverse order
  c2h::device_vector<value_t> expected_output{1, 2, 3, 4, 5, 10, 9, 8, 7, 6};
  c2h::device_vector<int> expected_num_selected{5};

  REQUIRE(d_num_selected == expected_num_selected);
  REQUIRE(d_out == expected_output);
}

C2H_TEST("Device partition flagged uses environment", "[partition][device]")
{
  using value_t     = int;
  using num_items_t = int;

  num_items_t num_items = 10;
  auto d_in             = c2h::device_vector<value_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto d_flags          = c2h::device_vector<char>{1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
  auto d_out            = c2h::device_vector<value_t>(num_items);
  auto d_num_selected   = c2h::device_vector<int>(1);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DevicePartition::Flagged(
      nullptr,
      expected_bytes_allocated,
      d_in.begin(),
      d_flags.begin(),
      d_out.begin(),
      d_num_selected.begin(),
      num_items));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)}; // temp storage size

  device_partition_flagged(d_in.begin(), d_flags.begin(), d_out.begin(), d_num_selected.begin(), num_items, env);

  // Flagged: {1, 3, 5, 7, 9} at front, unflagged: {2, 4, 6, 8, 10} at back in reverse order
  c2h::device_vector<value_t> expected_output{1, 3, 5, 7, 9, 10, 8, 6, 4, 2};
  c2h::device_vector<int> expected_num_selected{5};

  REQUIRE(d_num_selected == expected_num_selected);
  REQUIRE(d_out == expected_output);
}

TEST_CASE("Device partition uses custom stream", "[partition][device]")
{
  using value_t     = int;
  using num_items_t = int;

  num_items_t num_items = 8;
  auto d_in             = c2h::device_vector<value_t>{1, 2, 3, 4, 5, 6, 7, 8};
  auto d_out            = c2h::device_vector<value_t>(num_items);
  auto d_num_selected   = c2h::device_vector<int>(1);

  less_than_t<value_t> select_op{5};

  cudaStream_t custom_stream;
  REQUIRE(cudaSuccess == cudaStreamCreate(&custom_stream));

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DevicePartition::If(
      nullptr, expected_bytes_allocated, d_in.begin(), d_out.begin(), d_num_selected.begin(), num_items, select_op));

  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, cuda::stream_ref{custom_stream}};
  auto env         = stdexec::env{stream_prop, expected_allocation_size(expected_bytes_allocated)};

  device_partition_if(d_in.begin(), d_out.begin(), d_num_selected.begin(), num_items, select_op, env);

  REQUIRE(cudaSuccess == cudaStreamSynchronize(custom_stream));

  c2h::device_vector<value_t> expected_output{1, 2, 3, 4, 8, 7, 6, 5};
  c2h::device_vector<int> expected_num_selected{4};

  REQUIRE(d_num_selected == expected_num_selected);
  REQUIRE(d_out == expected_output);

  REQUIRE(cudaSuccess == cudaStreamDestroy(custom_stream));
}
