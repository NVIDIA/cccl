// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_run_length_encode.cuh>

#include <thrust/device_vector.h>

#include <cuda/devices>
#include <cuda/iterator>
#include <cuda/stream>

#include "catch2_test_env_launch_helper.h"

DECLARE_LAUNCH_WRAPPER(cub::DeviceRunLengthEncode::Encode, run_length_encode_env);
DECLARE_LAUNCH_WRAPPER(cub::DeviceRunLengthEncode::NonTrivialRuns, non_trivial_runs_env);

// %PARAM% TEST_LAUNCH lid 0:1:2

#include <c2h/catch2_test_helper.h>

namespace stdexec = cuda::std::execution;

#if TEST_LAUNCH == 0

TEST_CASE("DeviceRunLengthEncode::Encode works with default environment", "[run_length_encode][device]")
{
  auto d_in           = c2h::device_vector<int>{0, 2, 2, 9, 5, 5, 5, 8};
  auto d_unique_out   = c2h::device_vector<int>(8);
  auto d_counts_out   = c2h::device_vector<int>(8);
  auto d_num_runs_out = c2h::device_vector<int>(1);

  REQUIRE(cudaSuccess
          == cub::DeviceRunLengthEncode::Encode(
            d_in.begin(), d_unique_out.begin(), d_counts_out.begin(), d_num_runs_out.begin(), (int) d_in.size()));

  c2h::device_vector<int> expected_unique{0, 2, 9, 5, 8};
  c2h::device_vector<int> expected_counts{1, 2, 1, 3, 1};
  c2h::device_vector<int> expected_num_runs{5};

  REQUIRE(d_num_runs_out == expected_num_runs);
  d_unique_out.resize(d_num_runs_out[0]);
  d_counts_out.resize(d_num_runs_out[0]);
  REQUIRE(d_unique_out == expected_unique);
  REQUIRE(d_counts_out == expected_counts);
}

TEST_CASE("DeviceRunLengthEncode::NonTrivialRuns works with default environment", "[run_length_encode][device]")
{
  auto d_in           = c2h::device_vector<int>{0, 2, 2, 9, 5, 5, 5, 8};
  auto d_offsets_out  = c2h::device_vector<int>(8);
  auto d_lengths_out  = c2h::device_vector<int>(8);
  auto d_num_runs_out = c2h::device_vector<int>(1);

  REQUIRE(cudaSuccess
          == cub::DeviceRunLengthEncode::NonTrivialRuns(
            d_in.begin(), d_offsets_out.begin(), d_lengths_out.begin(), d_num_runs_out.begin(), (int) d_in.size()));

  c2h::device_vector<int> expected_offsets{1, 4};
  c2h::device_vector<int> expected_lengths{2, 3};
  c2h::device_vector<int> expected_num_runs{2};

  REQUIRE(d_num_runs_out == expected_num_runs);
  d_offsets_out.resize(d_num_runs_out[0]);
  d_lengths_out.resize(d_num_runs_out[0]);
  REQUIRE(d_offsets_out == expected_offsets);
  REQUIRE(d_lengths_out == expected_lengths);
}

#endif

C2H_TEST("DeviceRunLengthEncode::Encode uses environment", "[run_length_encode][device]")
{
  auto d_in           = c2h::device_vector<int>{1, 1, 1, 2, 2, 3, 4, 4, 4, 4};
  auto d_unique_out   = c2h::device_vector<int>(10);
  auto d_counts_out   = c2h::device_vector<int>(10);
  auto d_num_runs_out = c2h::device_vector<int>(1);
  int num_items       = static_cast<int>(d_in.size());

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceRunLengthEncode::Encode(
      nullptr,
      expected_bytes_allocated,
      d_in.begin(),
      d_unique_out.begin(),
      d_counts_out.begin(),
      d_num_runs_out.begin(),
      num_items));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  run_length_encode_env(
    d_in.begin(), d_unique_out.begin(), d_counts_out.begin(), d_num_runs_out.begin(), num_items, env);

  c2h::device_vector<int> expected_unique{1, 2, 3, 4};
  c2h::device_vector<int> expected_counts{3, 2, 1, 4};
  c2h::device_vector<int> expected_num_runs{4};

  REQUIRE(d_num_runs_out == expected_num_runs);
  d_unique_out.resize(d_num_runs_out[0]);
  d_counts_out.resize(d_num_runs_out[0]);
  REQUIRE(d_unique_out == expected_unique);
  REQUIRE(d_counts_out == expected_counts);
}

C2H_TEST("DeviceRunLengthEncode::NonTrivialRuns uses environment", "[run_length_encode][device]")
{
  auto d_in           = c2h::device_vector<int>{1, 1, 1, 2, 2, 3, 4, 4, 4, 4};
  auto d_offsets_out  = c2h::device_vector<int>(10);
  auto d_lengths_out  = c2h::device_vector<int>(10);
  auto d_num_runs_out = c2h::device_vector<int>(1);
  int num_items       = static_cast<int>(d_in.size());

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceRunLengthEncode::NonTrivialRuns(
      nullptr,
      expected_bytes_allocated,
      d_in.begin(),
      d_offsets_out.begin(),
      d_lengths_out.begin(),
      d_num_runs_out.begin(),
      num_items));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  non_trivial_runs_env(
    d_in.begin(), d_offsets_out.begin(), d_lengths_out.begin(), d_num_runs_out.begin(), num_items, env);

  c2h::device_vector<int> expected_offsets{0, 3, 6};
  c2h::device_vector<int> expected_lengths{3, 2, 4};
  c2h::device_vector<int> expected_num_runs{3};

  REQUIRE(d_num_runs_out == expected_num_runs);
  d_offsets_out.resize(d_num_runs_out[0]);
  d_lengths_out.resize(d_num_runs_out[0]);
  REQUIRE(d_offsets_out == expected_offsets);
  REQUIRE(d_lengths_out == expected_lengths);
}

TEST_CASE("DeviceRunLengthEncode::Encode uses custom stream", "[run_length_encode][device]")
{
  auto d_in           = c2h::device_vector<int>{0, 2, 2, 9, 5, 5, 5, 8};
  auto d_unique_out   = c2h::device_vector<int>(8);
  auto d_counts_out   = c2h::device_vector<int>(8);
  auto d_num_runs_out = c2h::device_vector<int>(1);
  int num_items       = static_cast<int>(d_in.size());

  cuda::stream custom_stream{cuda::devices[0]};

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceRunLengthEncode::Encode(
      nullptr,
      expected_bytes_allocated,
      d_in.begin(),
      d_unique_out.begin(),
      d_counts_out.begin(),
      d_num_runs_out.begin(),
      num_items));

  cuda::stream_ref stream_ref{custom_stream};
  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, stream_ref};
  auto env         = stdexec::env{stream_prop, expected_allocation_size(expected_bytes_allocated)};

  run_length_encode_env(
    d_in.begin(), d_unique_out.begin(), d_counts_out.begin(), d_num_runs_out.begin(), num_items, env);

  custom_stream.sync();

  c2h::device_vector<int> expected_unique{0, 2, 9, 5, 8};
  c2h::device_vector<int> expected_counts{1, 2, 1, 3, 1};
  c2h::device_vector<int> expected_num_runs{5};

  REQUIRE(d_num_runs_out == expected_num_runs);
  d_unique_out.resize(d_num_runs_out[0]);
  d_counts_out.resize(d_num_runs_out[0]);
  REQUIRE(d_unique_out == expected_unique);
  REQUIRE(d_counts_out == expected_counts);
}

TEST_CASE("DeviceRunLengthEncode::NonTrivialRuns uses custom stream", "[run_length_encode][device]")
{
  auto d_in           = c2h::device_vector<int>{0, 2, 2, 9, 5, 5, 5, 8};
  auto d_offsets_out  = c2h::device_vector<int>(8);
  auto d_lengths_out  = c2h::device_vector<int>(8);
  auto d_num_runs_out = c2h::device_vector<int>(1);
  int num_items       = static_cast<int>(d_in.size());

  cuda::stream custom_stream{cuda::devices[0]};

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceRunLengthEncode::NonTrivialRuns(
      nullptr,
      expected_bytes_allocated,
      d_in.begin(),
      d_offsets_out.begin(),
      d_lengths_out.begin(),
      d_num_runs_out.begin(),
      num_items));

  cuda::stream_ref stream_ref{custom_stream};
  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, stream_ref};
  auto env         = stdexec::env{stream_prop, expected_allocation_size(expected_bytes_allocated)};

  non_trivial_runs_env(
    d_in.begin(), d_offsets_out.begin(), d_lengths_out.begin(), d_num_runs_out.begin(), num_items, env);

  custom_stream.sync();

  c2h::device_vector<int> expected_offsets{1, 4};
  c2h::device_vector<int> expected_lengths{2, 3};
  c2h::device_vector<int> expected_num_runs{2};

  REQUIRE(d_num_runs_out == expected_num_runs);
  d_offsets_out.resize(d_num_runs_out[0]);
  d_lengths_out.resize(d_num_runs_out[0]);
  REQUIRE(d_offsets_out == expected_offsets);
  REQUIRE(d_lengths_out == expected_lengths);
}
