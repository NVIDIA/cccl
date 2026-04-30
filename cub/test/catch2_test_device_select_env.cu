// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_select.cuh>

#include <thrust/device_vector.h>

#include <cuda/__device/compute_capability.h>
#include <cuda/__iterator/constant_iterator.h>
#include <cuda/iterator>

#include "catch2_test_device_select_common.cuh"
#include "catch2_test_env_launch_helper.h"

DECLARE_LAUNCH_WRAPPER(cub::DeviceSelect::If, device_select_if);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSelect::Flagged, device_select_flagged);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSelect::FlaggedIf, device_select_flagged_if);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSelect::Unique, device_select_unique);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSelect::UniqueByKey, device_select_unique_by_key);

// %PARAM% TEST_LAUNCH lid 0:1:2

#include <cuda/__execution/require.h>

#include <c2h/catch2_test_helper.h>

namespace stdexec = cuda::std::execution;

#if TEST_LAUNCH == 0

using block_size_check_t = block_size_extracting_op<cuda::std::equal_to<>>;

TEST_CASE("Device select works with default environment", "[select][device]")
{
  using value_t     = int;
  using num_items_t = int;

  num_items_t num_items = 8;
  auto d_in             = c2h::device_vector<value_t>{1, 2, 3, 4, 5, 6, 7, 8};
  auto d_out            = c2h::device_vector<value_t>(num_items);
  auto d_num_selected   = c2h::device_vector<unsigned int>(1);

  less_than_t<value_t> select_op{5};

  // launch wrapper always assumes the last argument is the environment
  REQUIRE(
    cudaSuccess == cub::DeviceSelect::If(d_in.begin(), d_out.begin(), d_num_selected.begin(), num_items, select_op));

  c2h::device_vector<value_t> expected_output{1, 2, 3, 4};
  c2h::device_vector<int> expected_num_selected{4};

  REQUIRE(d_num_selected == expected_num_selected);
  d_out.resize(d_num_selected[0]);
  REQUIRE(d_out == expected_output);
}

TEST_CASE("Device select flagged works with default environment", "[select][device]")
{
  using value_t     = int;
  using num_items_t = int;

  num_items_t num_items = 8;
  auto d_in             = c2h::device_vector<value_t>{1, 2, 3, 4, 5, 6, 7, 8};
  auto d_flags          = c2h::device_vector<char>{1, 0, 0, 1, 0, 1, 1, 0};
  auto d_out            = c2h::device_vector<value_t>(num_items);
  auto d_num_selected   = c2h::device_vector<unsigned int>(1);

  // launch wrapper always assumes the last argument is the environment
  REQUIRE(
    cudaSuccess
    == cub::DeviceSelect::Flagged(d_in.begin(), d_flags.begin(), d_out.begin(), d_num_selected.begin(), num_items));

  c2h::device_vector<value_t> expected_output{1, 4, 6, 7};
  c2h::device_vector<int> expected_num_selected{4};

  REQUIRE(d_num_selected == expected_num_selected);
  d_out.resize(d_num_selected[0]);
  REQUIRE(d_out == expected_output);
}

TEST_CASE("Device select flagged_if works with default environment", "[select][device]")
{
  using value_t     = int;
  using num_items_t = int;

  num_items_t num_items = 8;
  auto d_in             = c2h::device_vector<value_t>{1, 2, 3, 4, 5, 6, 7, 8};
  auto d_flags          = c2h::device_vector<int>{2, 1, 1, 4, 1, 6, 6, 1};
  auto d_out            = c2h::device_vector<value_t>(num_items);
  auto d_num_selected   = c2h::device_vector<unsigned int>(1);

  mod_n<int> select_op{2};

  REQUIRE(cudaSuccess
          == cub::DeviceSelect::FlaggedIf(
            d_in.begin(), d_flags.begin(), d_out.begin(), d_num_selected.begin(), num_items, select_op));

  c2h::device_vector<value_t> expected_output{1, 4, 6, 7};
  c2h::device_vector<int> expected_num_selected{4};

  REQUIRE(d_num_selected == expected_num_selected);
  d_out.resize(d_num_selected[0]);
  REQUIRE(d_out == expected_output);
}

TEST_CASE("Device select flagged in-place works with default environment", "[select][device]")
{
  using value_t     = int;
  using num_items_t = int;

  num_items_t num_items = 8;
  auto d_data           = c2h::device_vector<value_t>{1, 2, 3, 4, 5, 6, 7, 8};
  auto d_flags          = c2h::device_vector<char>{1, 0, 0, 1, 0, 1, 1, 0};
  auto d_num_selected   = c2h::device_vector<unsigned int>(1);

  REQUIRE(
    cudaSuccess == cub::DeviceSelect::Flagged(d_data.begin(), d_flags.begin(), d_num_selected.begin(), num_items));

  c2h::device_vector<value_t> expected_output{1, 4, 6, 7};
  c2h::device_vector<int> expected_num_selected{4};

  REQUIRE(d_num_selected == expected_num_selected);
  d_data.resize(d_num_selected[0]);
  REQUIRE(d_data == expected_output);
}

TEST_CASE("Device select if in-place works with default environment", "[select][device]")
{
  using value_t     = int;
  using num_items_t = int;

  num_items_t num_items = 8;
  auto d_data           = c2h::device_vector<value_t>{1, 2, 3, 4, 5, 6, 7, 8};
  auto d_num_selected   = c2h::device_vector<unsigned int>(1);

  less_than_t<value_t> select_op{5};

  REQUIRE(cudaSuccess == cub::DeviceSelect::If(d_data.begin(), d_num_selected.begin(), num_items, select_op));

  c2h::device_vector<value_t> expected_output{1, 2, 3, 4};
  c2h::device_vector<int> expected_num_selected{4};

  REQUIRE(d_num_selected == expected_num_selected);
  d_data.resize(d_num_selected[0]);
  REQUIRE(d_data == expected_output);
}

TEST_CASE("Device select flagged_if in-place works with default environment", "[select][device]")
{
  using value_t     = int;
  using num_items_t = int;

  num_items_t num_items = 8;
  auto d_data           = c2h::device_vector<value_t>{1, 2, 3, 4, 5, 6, 7, 8};
  auto d_flags          = c2h::device_vector<int>{2, 1, 1, 4, 1, 6, 6, 1};
  auto d_num_selected   = c2h::device_vector<unsigned int>(1);

  mod_n<int> select_op{2};

  REQUIRE(
    cudaSuccess
    == cub::DeviceSelect::FlaggedIf(d_data.begin(), d_flags.begin(), d_num_selected.begin(), num_items, select_op));

  c2h::device_vector<value_t> expected_output{1, 4, 6, 7};
  c2h::device_vector<int> expected_num_selected{4};

  REQUIRE(d_num_selected == expected_num_selected);
  d_data.resize(d_num_selected[0]);
  REQUIRE(d_data == expected_output);
}

TEST_CASE("Device select unique works with default environment", "[select][device]")
{
  using value_t     = int;
  using num_items_t = int;

  num_items_t num_items = 8;
  auto d_in             = c2h::device_vector<value_t>{0, 2, 2, 9, 5, 5, 5, 8};
  auto d_out            = c2h::device_vector<value_t>(num_items);
  auto d_num_selected   = c2h::device_vector<unsigned int>(1);

  REQUIRE(cudaSuccess == cub::DeviceSelect::Unique(d_in.begin(), d_out.begin(), d_num_selected.begin(), num_items));

  c2h::device_vector<value_t> expected_output{0, 2, 9, 5, 8};
  c2h::device_vector<int> expected_num_selected{5};

  REQUIRE(d_num_selected == expected_num_selected);
  d_out.resize(d_num_selected[0]);
  REQUIRE(d_out == expected_output);
}

TEST_CASE("Device select unique with custom equality_op works with default environment", "[select][device]")
{
  using value_t     = int;
  using num_items_t = int;

  num_items_t num_items = 8;
  auto d_in             = c2h::device_vector<value_t>{0, 3, 6, 1, 4, 7, 2, 5};
  auto d_out            = c2h::device_vector<value_t>(num_items);
  auto d_num_selected   = c2h::device_vector<unsigned int>(1);

  eq_mod3_t<value_t> eq_mod3{};

  REQUIRE(
    cudaSuccess == cub::DeviceSelect::Unique(d_in.begin(), d_out.begin(), d_num_selected.begin(), num_items, eq_mod3));

  c2h::device_vector<value_t> expected_output{0, 1, 2};
  c2h::device_vector<int> expected_num_selected{3};

  REQUIRE(d_num_selected == expected_num_selected);
  d_out.resize(d_num_selected[0]);
  REQUIRE(d_out == expected_output);
}

TEST_CASE("Device select unique_by_key works with default environment", "[select][device]")
{
  using value_t     = int;
  using num_items_t = int;

  num_items_t num_items = 8;
  auto d_keys_in        = c2h::device_vector<value_t>{0, 2, 2, 9, 5, 5, 5, 8};
  auto d_values_in      = c2h::device_vector<value_t>{1, 2, 3, 4, 5, 6, 7, 8};
  auto d_keys_out       = c2h::device_vector<value_t>(num_items);
  auto d_values_out     = c2h::device_vector<value_t>(num_items);
  auto d_num_selected   = c2h::device_vector<unsigned int>(1);

  REQUIRE(
    cudaSuccess
    == cub::DeviceSelect::UniqueByKey(
      d_keys_in.begin(),
      d_values_in.begin(),
      d_keys_out.begin(),
      d_values_out.begin(),
      d_num_selected.begin(),
      num_items));

  c2h::device_vector<value_t> expected_keys{0, 2, 9, 5, 8};
  c2h::device_vector<value_t> expected_values{1, 2, 4, 5, 8};
  c2h::device_vector<int> expected_num_selected{5};

  REQUIRE(d_num_selected == expected_num_selected);
  d_keys_out.resize(d_num_selected[0]);
  d_values_out.resize(d_num_selected[0]);
  REQUIRE(d_keys_out == expected_keys);
  REQUIRE(d_values_out == expected_values);
}

TEST_CASE("Device select unique_by_key works with default environment and explicit env", "[select][device]")
{
  using value_t     = int;
  using num_items_t = int;

  num_items_t num_items = 8;
  auto d_keys_in        = c2h::device_vector<value_t>{0, 2, 2, 9, 5, 5, 5, 8};
  auto d_values_in      = c2h::device_vector<value_t>{1, 2, 3, 4, 5, 6, 7, 8};
  auto d_keys_out       = c2h::device_vector<value_t>(num_items);
  auto d_values_out     = c2h::device_vector<value_t>(num_items);
  auto d_num_selected   = c2h::device_vector<int>(1);

  auto env = stdexec::env{};

  REQUIRE(
    cudaSuccess
    == cub::DeviceSelect::UniqueByKey(
      d_keys_in.begin(),
      d_values_in.begin(),
      d_keys_out.begin(),
      d_values_out.begin(),
      d_num_selected.begin(),
      num_items,
      env));

  c2h::device_vector<value_t> expected_keys{0, 2, 9, 5, 8};
  c2h::device_vector<value_t> expected_values{1, 2, 4, 5, 8};
  c2h::device_vector<int> expected_num_selected{5};

  REQUIRE(d_num_selected == expected_num_selected);
  d_keys_out.resize(d_num_selected[0]);
  d_values_out.resize(d_num_selected[0]);
  REQUIRE(d_keys_out == expected_keys);
  REQUIRE(d_values_out == expected_values);
}

TEST_CASE("Device select unique_by_key default tuning chooses target block size", "[select][device]")
{
  using num_items_t = int;
  using key_t       = int;
  using value_t     = int;

  using selector_t = cub::detail::unique_by_key::policy_selector_from_types<key_t, value_t>;

  int current_device{};
  REQUIRE(cudaSuccess == cudaGetDevice(&current_device));

  cuda::compute_capability cc{};
  REQUIRE(cudaSuccess == cub::detail::ptx_compute_cap(cc, current_device));

  const auto target_block_size = selector_t{}(cc).block_threads;

  num_items_t num_items = 1;
  auto d_keys_in        = c2h::device_vector<key_t>{0};
  auto d_keys_out       = c2h::device_vector<key_t>(1);
  auto d_values_out     = c2h::device_vector<value_t>(1);
  auto d_num_selected   = c2h::device_vector<unsigned int>(1);
  auto d_block_size     = c2h::device_vector<unsigned int>(1);
  block_size_check_t equality_op{thrust::raw_pointer_cast(d_block_size.data())};
  auto d_values_in = cuda::constant_iterator(value_t{1});

  REQUIRE(
    cudaSuccess
    == cub::DeviceSelect::UniqueByKey(
      d_keys_in.begin(),
      d_values_in,
      d_keys_out.begin(),
      d_values_out.begin(),
      d_num_selected.begin(),
      num_items,
      equality_op));

  REQUIRE(d_num_selected[0] == 1);
  REQUIRE(d_keys_out[0] == key_t{0});
  REQUIRE(d_values_out[0] == value_t{1});
  REQUIRE(d_block_size[0] == static_cast<unsigned int>(target_block_size));
}

#endif

C2H_TEST("Device select uses environment", "[select][device]")
{
  using value_t     = int;
  using num_items_t = int;

  num_items_t num_items = 10;
  auto d_in             = c2h::device_vector<value_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto d_out            = c2h::device_vector<value_t>(num_items);
  auto d_num_selected   = c2h::device_vector<unsigned int>(1);

  less_than_t<value_t> select_op{6};

  size_t expected_bytes_allocated{};
  // calculate expected_bytes_allocated - call CUB API directly, not through wrapper
  REQUIRE(
    cudaSuccess
    == cub::DeviceSelect::If(
      nullptr, expected_bytes_allocated, d_in.begin(), d_out.begin(), d_num_selected.begin(), num_items, select_op));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)}; // temp storage size

  device_select_if(d_in.begin(), d_out.begin(), d_num_selected.begin(), num_items, select_op, env);

  c2h::device_vector<value_t> expected_output{1, 2, 3, 4, 5};
  c2h::device_vector<int> expected_num_selected{5};

  REQUIRE(d_num_selected == expected_num_selected);
  d_out.resize(d_num_selected[0]);
  REQUIRE(d_out == expected_output);
}

C2H_TEST("Device select flagged uses environment", "[select][device]")
{
  using value_t     = int;
  using num_items_t = int;

  num_items_t num_items = 10;
  auto d_in             = c2h::device_vector<value_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto d_flags          = c2h::device_vector<char>{1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
  auto d_out            = c2h::device_vector<value_t>(num_items);
  auto d_num_selected   = c2h::device_vector<unsigned int>(1);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSelect::Flagged(
      nullptr,
      expected_bytes_allocated,
      d_in.begin(),
      d_flags.begin(),
      d_out.begin(),
      d_num_selected.begin(),
      num_items));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)}; // temp storage size

  device_select_flagged(d_in.begin(), d_flags.begin(), d_out.begin(), d_num_selected.begin(), num_items, env);

  c2h::device_vector<value_t> expected_output{1, 3, 5, 7, 9};
  c2h::device_vector<int> expected_num_selected{5};

  REQUIRE(d_num_selected == expected_num_selected);
  d_out.resize(d_num_selected[0]);
  REQUIRE(d_out == expected_output);
}

C2H_TEST("Device select flagged_if uses environment", "[select][device]")
{
  using value_t     = int;
  using num_items_t = int;

  num_items_t num_items = 10;
  auto d_in             = c2h::device_vector<value_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto d_flags          = c2h::device_vector<int>{2, 1, 2, 1, 2, 1, 2, 1, 2, 1};
  auto d_out            = c2h::device_vector<value_t>(num_items);
  auto d_num_selected   = c2h::device_vector<unsigned int>(1);

  mod_n<int> select_op{2};

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSelect::FlaggedIf(
      nullptr,
      expected_bytes_allocated,
      d_in.begin(),
      d_flags.begin(),
      d_out.begin(),
      d_num_selected.begin(),
      num_items,
      select_op));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_select_flagged_if(
    d_in.begin(), d_flags.begin(), d_out.begin(), d_num_selected.begin(), num_items, select_op, env);

  c2h::device_vector<value_t> expected_output{1, 3, 5, 7, 9};
  c2h::device_vector<int> expected_num_selected{5};

  REQUIRE(d_num_selected == expected_num_selected);
  d_out.resize(d_num_selected[0]);
  REQUIRE(d_out == expected_output);
}

C2H_TEST("Device select flagged in-place uses environment", "[select][device]")
{
  using value_t     = int;
  using num_items_t = int;

  num_items_t num_items = 10;
  auto d_data           = c2h::device_vector<value_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto d_flags          = c2h::device_vector<char>{1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
  auto d_num_selected   = c2h::device_vector<unsigned int>(1);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSelect::Flagged(
      nullptr,
      expected_bytes_allocated,
      d_data.begin(),
      d_flags.begin(),
      d_data.begin(),
      d_num_selected.begin(),
      num_items));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  REQUIRE(
    cudaSuccess == cub::DeviceSelect::Flagged(d_data.begin(), d_flags.begin(), d_num_selected.begin(), num_items, env));

  c2h::device_vector<value_t> expected_output{1, 3, 5, 7, 9};
  c2h::device_vector<int> expected_num_selected{5};

  REQUIRE(d_num_selected == expected_num_selected);
  d_data.resize(d_num_selected[0]);
  REQUIRE(d_data == expected_output);
}

C2H_TEST("Device select if in-place uses environment", "[select][device]")
{
  using value_t     = int;
  using num_items_t = int;

  num_items_t num_items = 10;
  auto d_data           = c2h::device_vector<value_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto d_num_selected   = c2h::device_vector<unsigned int>(1);

  less_than_t<value_t> select_op{6};

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSelect::If(
      nullptr, expected_bytes_allocated, d_data.begin(), d_data.begin(), d_num_selected.begin(), num_items, select_op));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  REQUIRE(cudaSuccess == cub::DeviceSelect::If(d_data.begin(), d_num_selected.begin(), num_items, select_op, env));

  c2h::device_vector<value_t> expected_output{1, 2, 3, 4, 5};
  c2h::device_vector<int> expected_num_selected{5};

  REQUIRE(d_num_selected == expected_num_selected);
  d_data.resize(d_num_selected[0]);
  REQUIRE(d_data == expected_output);
}

C2H_TEST("Device select flagged_if in-place uses environment", "[select][device]")
{
  using value_t     = int;
  using num_items_t = int;

  num_items_t num_items = 10;
  auto d_data           = c2h::device_vector<value_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto d_flags          = c2h::device_vector<int>{2, 1, 2, 1, 2, 1, 2, 1, 2, 1};
  auto d_num_selected   = c2h::device_vector<unsigned int>(1);

  mod_n<int> select_op{2};

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSelect::FlaggedIf(
      nullptr,
      expected_bytes_allocated,
      d_data.begin(),
      d_flags.begin(),
      d_data.begin(),
      d_num_selected.begin(),
      num_items,
      select_op));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  REQUIRE(cudaSuccess
          == cub::DeviceSelect::FlaggedIf(
            d_data.begin(), d_flags.begin(), d_num_selected.begin(), num_items, select_op, env));

  c2h::device_vector<value_t> expected_output{1, 3, 5, 7, 9};
  c2h::device_vector<int> expected_num_selected{5};

  REQUIRE(d_num_selected == expected_num_selected);
  d_data.resize(d_num_selected[0]);
  REQUIRE(d_data == expected_output);
}

C2H_TEST("Device select unique uses environment", "[select][device]")
{
  using value_t     = int;
  using num_items_t = int;

  num_items_t num_items = 10;
  auto d_in             = c2h::device_vector<value_t>{1, 1, 2, 2, 3, 3, 4, 4, 5, 5};
  auto d_out            = c2h::device_vector<value_t>(num_items);
  auto d_num_selected   = c2h::device_vector<unsigned int>(1);

  size_t expected_bytes_allocated{};
  REQUIRE(cudaSuccess
          == cub::DeviceSelect::Unique(
            nullptr, expected_bytes_allocated, d_in.begin(), d_out.begin(), d_num_selected.begin(), num_items));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_select_unique(d_in.begin(), d_out.begin(), d_num_selected.begin(), num_items, env);

  c2h::device_vector<value_t> expected_output{1, 2, 3, 4, 5};
  c2h::device_vector<int> expected_num_selected{5};

  REQUIRE(d_num_selected == expected_num_selected);
  d_out.resize(d_num_selected[0]);
  REQUIRE(d_out == expected_output);
}

C2H_TEST("Device select unique with custom equality_op uses environment", "[select][device]")
{
  using value_t     = int;
  using num_items_t = int;

  num_items_t num_items = 8;
  auto d_in             = c2h::device_vector<value_t>{0, 3, 6, 1, 4, 7, 2, 5};
  auto d_out            = c2h::device_vector<value_t>(num_items);
  auto d_num_selected   = c2h::device_vector<unsigned int>(1);

  eq_mod3_t<value_t> eq_mod3{};

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSelect::Unique(
      nullptr, expected_bytes_allocated, d_in.begin(), d_out.begin(), d_num_selected.begin(), num_items, eq_mod3));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_select_unique(d_in.begin(), d_out.begin(), d_num_selected.begin(), num_items, eq_mod3, env);

  c2h::device_vector<value_t> expected_output{0, 1, 2};
  c2h::device_vector<int> expected_num_selected{3};

  REQUIRE(d_num_selected == expected_num_selected);
  d_out.resize(d_num_selected[0]);
  REQUIRE(d_out == expected_output);
}

C2H_TEST("Device select unique_by_key uses environment", "[select][device]")
{
  using value_t     = int;
  using num_items_t = int;

  num_items_t num_items = 10;
  auto d_keys_in        = c2h::device_vector<value_t>{1, 1, 2, 2, 3, 3, 4, 4, 5, 5};
  auto d_values_in      = c2h::device_vector<value_t>{10, 11, 20, 21, 30, 31, 40, 41, 50, 51};
  auto d_keys_out       = c2h::device_vector<value_t>(num_items);
  auto d_values_out     = c2h::device_vector<value_t>(num_items);
  auto d_num_selected   = c2h::device_vector<unsigned int>(1);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSelect::UniqueByKey(
      nullptr,
      expected_bytes_allocated,
      d_keys_in.begin(),
      d_values_in.begin(),
      d_keys_out.begin(),
      d_values_out.begin(),
      d_num_selected.begin(),
      num_items));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_select_unique_by_key(
    d_keys_in.begin(),
    d_values_in.begin(),
    d_keys_out.begin(),
    d_values_out.begin(),
    d_num_selected.begin(),
    num_items,
    ::cuda::std::equal_to<>{},
    env);

  c2h::device_vector<value_t> expected_keys{1, 2, 3, 4, 5};
  c2h::device_vector<value_t> expected_values{10, 20, 30, 40, 50};
  c2h::device_vector<int> expected_num_selected{5};

  REQUIRE(d_num_selected == expected_num_selected);
  d_keys_out.resize(d_num_selected[0]);
  d_values_out.resize(d_num_selected[0]);
  REQUIRE(d_keys_out == expected_keys);
  REQUIRE(d_values_out == expected_values);
}

C2H_TEST("Device select unique_by_key uses environment without equality_op", "[select][device]")
{
  using value_t     = int;
  using num_items_t = int;

  num_items_t num_items = 10;
  auto d_keys_in        = c2h::device_vector<value_t>{1, 1, 2, 2, 3, 3, 4, 4, 5, 5};
  auto d_values_in      = c2h::device_vector<value_t>{10, 11, 20, 21, 30, 31, 40, 41, 50, 51};
  auto d_keys_out       = c2h::device_vector<value_t>(num_items);
  auto d_values_out     = c2h::device_vector<value_t>(num_items);
  auto d_num_selected   = c2h::device_vector<int>(1);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSelect::UniqueByKey(
      nullptr,
      expected_bytes_allocated,
      d_keys_in.begin(),
      d_values_in.begin(),
      d_keys_out.begin(),
      d_values_out.begin(),
      d_num_selected.begin(),
      num_items));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_select_unique_by_key(
    d_keys_in.begin(),
    d_values_in.begin(),
    d_keys_out.begin(),
    d_values_out.begin(),
    d_num_selected.begin(),
    num_items,
    env);

  c2h::device_vector<value_t> expected_keys{1, 2, 3, 4, 5};
  c2h::device_vector<value_t> expected_values{10, 20, 30, 40, 50};
  c2h::device_vector<int> expected_num_selected{5};

  REQUIRE(d_num_selected == expected_num_selected);
  d_keys_out.resize(d_num_selected[0]);
  d_values_out.resize(d_num_selected[0]);
  REQUIRE(d_keys_out == expected_keys);
  REQUIRE(d_values_out == expected_values);
}

TEST_CASE("Device select uses custom stream", "[select][device]")
{
  using value_t     = int;
  using num_items_t = int;

  num_items_t num_items = 8;
  auto d_in             = c2h::device_vector<value_t>{1, 2, 3, 4, 5, 6, 7, 8};
  auto d_out            = c2h::device_vector<value_t>(num_items);
  auto d_num_selected   = c2h::device_vector<unsigned int>(1);

  less_than_t<value_t> select_op{5};

  cudaStream_t custom_stream;
  REQUIRE(cudaSuccess == cudaStreamCreate(&custom_stream));

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSelect::If(
      nullptr, expected_bytes_allocated, d_in.begin(), d_out.begin(), d_num_selected.begin(), num_items, select_op));

  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, cuda::stream_ref{custom_stream}};
  auto env         = stdexec::env{stream_prop, expected_allocation_size(expected_bytes_allocated)};

  device_select_if(d_in.begin(), d_out.begin(), d_num_selected.begin(), num_items, select_op, env);

  REQUIRE(cudaSuccess == cudaStreamSynchronize(custom_stream));

  c2h::device_vector<value_t> expected_output{1, 2, 3, 4};
  c2h::device_vector<int> expected_num_selected{4};

  REQUIRE(d_num_selected == expected_num_selected);
  d_out.resize(d_num_selected[0]);
  REQUIRE(d_out == expected_output);

  REQUIRE(cudaSuccess == cudaStreamDestroy(custom_stream));
}

TEST_CASE("Device select flagged uses custom stream", "[select][device]")
{
  using value_t     = int;
  using num_items_t = int;

  num_items_t num_items = 8;
  auto d_in             = c2h::device_vector<value_t>{1, 2, 3, 4, 5, 6, 7, 8};
  auto d_flags          = c2h::device_vector<char>{1, 0, 0, 1, 0, 1, 1, 0};
  auto d_out            = c2h::device_vector<value_t>(num_items);
  auto d_num_selected   = c2h::device_vector<unsigned int>(1);

  cudaStream_t custom_stream;
  REQUIRE(cudaSuccess == cudaStreamCreate(&custom_stream));

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSelect::Flagged(
      nullptr,
      expected_bytes_allocated,
      d_in.begin(),
      d_flags.begin(),
      d_out.begin(),
      d_num_selected.begin(),
      num_items));

  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, cuda::stream_ref{custom_stream}};
  auto env         = stdexec::env{stream_prop, expected_allocation_size(expected_bytes_allocated)};

  device_select_flagged(d_in.begin(), d_flags.begin(), d_out.begin(), d_num_selected.begin(), num_items, env);

  REQUIRE(cudaSuccess == cudaStreamSynchronize(custom_stream));

  c2h::device_vector<value_t> expected_output{1, 4, 6, 7};
  c2h::device_vector<int> expected_num_selected{4};

  REQUIRE(d_num_selected == expected_num_selected);
  d_out.resize(d_num_selected[0]);
  REQUIRE(d_out == expected_output);

  REQUIRE(cudaSuccess == cudaStreamDestroy(custom_stream));
}

TEST_CASE("Device select flagged_if uses custom stream", "[select][device]")
{
  using value_t     = int;
  using num_items_t = int;

  num_items_t num_items = 8;
  auto d_in             = c2h::device_vector<value_t>{1, 2, 3, 4, 5, 6, 7, 8};
  auto d_flags          = c2h::device_vector<int>{2, 1, 1, 4, 1, 6, 6, 1};
  auto d_out            = c2h::device_vector<value_t>(num_items);
  auto d_num_selected   = c2h::device_vector<unsigned int>(1);

  mod_n<int> select_op{2};

  cudaStream_t custom_stream;
  REQUIRE(cudaSuccess == cudaStreamCreate(&custom_stream));

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSelect::FlaggedIf(
      nullptr,
      expected_bytes_allocated,
      d_in.begin(),
      d_flags.begin(),
      d_out.begin(),
      d_num_selected.begin(),
      num_items,
      select_op));

  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, cuda::stream_ref{custom_stream}};
  auto env         = stdexec::env{stream_prop, expected_allocation_size(expected_bytes_allocated)};

  device_select_flagged_if(
    d_in.begin(), d_flags.begin(), d_out.begin(), d_num_selected.begin(), num_items, select_op, env);

  REQUIRE(cudaSuccess == cudaStreamSynchronize(custom_stream));

  c2h::device_vector<value_t> expected_output{1, 4, 6, 7};
  c2h::device_vector<int> expected_num_selected{4};

  REQUIRE(d_num_selected == expected_num_selected);
  d_out.resize(d_num_selected[0]);
  REQUIRE(d_out == expected_output);

  REQUIRE(cudaSuccess == cudaStreamDestroy(custom_stream));
}

TEST_CASE("Device select unique uses custom stream", "[select][device]")
{
  using value_t     = int;
  using num_items_t = int;

  num_items_t num_items = 8;
  auto d_in             = c2h::device_vector<value_t>{0, 2, 2, 9, 5, 5, 5, 8};
  auto d_out            = c2h::device_vector<value_t>(num_items);
  auto d_num_selected   = c2h::device_vector<unsigned int>(1);

  cudaStream_t custom_stream;
  REQUIRE(cudaSuccess == cudaStreamCreate(&custom_stream));

  size_t expected_bytes_allocated{};
  REQUIRE(cudaSuccess
          == cub::DeviceSelect::Unique(
            nullptr, expected_bytes_allocated, d_in.begin(), d_out.begin(), d_num_selected.begin(), num_items));

  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, cuda::stream_ref{custom_stream}};
  auto env         = stdexec::env{stream_prop, expected_allocation_size(expected_bytes_allocated)};

  device_select_unique(d_in.begin(), d_out.begin(), d_num_selected.begin(), num_items, env);

  REQUIRE(cudaSuccess == cudaStreamSynchronize(custom_stream));

  c2h::device_vector<value_t> expected_output{0, 2, 9, 5, 8};
  c2h::device_vector<int> expected_num_selected{5};

  REQUIRE(d_num_selected == expected_num_selected);
  d_out.resize(d_num_selected[0]);
  REQUIRE(d_out == expected_output);

  REQUIRE(cudaSuccess == cudaStreamDestroy(custom_stream));
}

TEST_CASE("Device select unique_by_key uses custom stream", "[select][device]")
{
  using value_t     = int;
  using num_items_t = int;

  num_items_t num_items = 8;
  auto d_keys_in        = c2h::device_vector<value_t>{0, 2, 2, 9, 5, 5, 5, 8};
  auto d_values_in      = c2h::device_vector<value_t>{1, 2, 3, 4, 5, 6, 7, 8};
  auto d_keys_out       = c2h::device_vector<value_t>(num_items);
  auto d_values_out     = c2h::device_vector<value_t>(num_items);
  auto d_num_selected   = c2h::device_vector<unsigned int>(1);

  cudaStream_t custom_stream;
  REQUIRE(cudaSuccess == cudaStreamCreate(&custom_stream));

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSelect::UniqueByKey(
      nullptr,
      expected_bytes_allocated,
      d_keys_in.begin(),
      d_values_in.begin(),
      d_keys_out.begin(),
      d_values_out.begin(),
      d_num_selected.begin(),
      num_items));

  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, cuda::stream_ref{custom_stream}};
  auto env         = stdexec::env{stream_prop, expected_allocation_size(expected_bytes_allocated)};

  device_select_unique_by_key(
    d_keys_in.begin(),
    d_values_in.begin(),
    d_keys_out.begin(),
    d_values_out.begin(),
    d_num_selected.begin(),
    num_items,
    ::cuda::std::equal_to<>{},
    env);

  REQUIRE(cudaSuccess == cudaStreamSynchronize(custom_stream));

  c2h::device_vector<value_t> expected_keys{0, 2, 9, 5, 8};
  c2h::device_vector<value_t> expected_values{1, 2, 4, 5, 8};
  c2h::device_vector<int> expected_num_selected{5};

  REQUIRE(d_num_selected == expected_num_selected);
  d_keys_out.resize(d_num_selected[0]);
  d_values_out.resize(d_num_selected[0]);
  REQUIRE(d_keys_out == expected_keys);
  REQUIRE(d_values_out == expected_values);

  REQUIRE(cudaSuccess == cudaStreamDestroy(custom_stream));
}
