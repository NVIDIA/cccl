// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_segmented_reduce.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "catch2_test_env_launch_helper.h"

DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedReduce::Reduce, device_segmented_reduce);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedReduce::Sum, device_segmented_reduce_sum);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedReduce::Min, device_segmented_reduce_min);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedReduce::Max, device_segmented_reduce_max);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedReduce::ArgMin, device_segmented_reduce_argmin);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedReduce::ArgMax, device_segmented_reduce_argmax);

// %PARAM% TEST_LAUNCH lid 0:1:2

#include <cuda/__execution/require.h>

#include <c2h/catch2_test_helper.h>

namespace stdexec = cuda::std::execution;

// Test data: 3 segments: {8,6,7,5, 3,0,9, 1,2} with offsets {0,4,7,9}

#if TEST_LAUNCH == 0

TEST_CASE("Device segmented reduce works with default environment", "[segmented_reduce][device]")
{
  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(3);

  REQUIRE(cudaSuccess
          == cub::DeviceSegmentedReduce::Reduce(
            d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, ::cuda::std::plus<>{}, 0));

  thrust::device_vector<int> expected{26, 12, 3};
  REQUIRE(d_out == expected);
}

TEST_CASE("Device segmented sum works with default environment", "[segmented_reduce][device]")
{
  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(3);

  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedReduce::Sum(d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1));

  thrust::device_vector<int> expected{26, 12, 3};
  REQUIRE(d_out == expected);
}

TEST_CASE("Device segmented min works with default environment", "[segmented_reduce][device]")
{
  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(3);

  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedReduce::Min(d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1));

  thrust::device_vector<int> expected{5, 0, 1};
  REQUIRE(d_out == expected);
}

TEST_CASE("Device segmented max works with default environment", "[segmented_reduce][device]")
{
  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(3);

  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedReduce::Max(d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1));

  thrust::device_vector<int> expected{8, 9, 2};
  REQUIRE(d_out == expected);
}

TEST_CASE("Device segmented argmin works with default environment", "[segmented_reduce][device]")
{
  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<cub::KeyValuePair<int, int>> d_out(3);

  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedReduce::ArgMin(d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1));

  thrust::host_vector<cub::KeyValuePair<int, int>> h_out(d_out);
  REQUIRE(h_out[0].key == 3);
  REQUIRE(h_out[0].value == 5);
  REQUIRE(h_out[1].key == 1);
  REQUIRE(h_out[1].value == 0);
  REQUIRE(h_out[2].key == 0);
  REQUIRE(h_out[2].value == 1);
}

TEST_CASE("Device segmented argmax works with default environment", "[segmented_reduce][device]")
{
  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<cub::KeyValuePair<int, int>> d_out(3);

  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedReduce::ArgMax(d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1));

  thrust::host_vector<cub::KeyValuePair<int, int>> h_out(d_out);
  REQUIRE(h_out[0].key == 0);
  REQUIRE(h_out[0].value == 8);
  REQUIRE(h_out[1].key == 2);
  REQUIRE(h_out[1].value == 9);
  REQUIRE(h_out[2].key == 1);
  REQUIRE(h_out[2].value == 2);
}

TEST_CASE("Device fixed-size segmented reduce works with default environment", "[segmented_reduce][device]")
{
  int num_segments = 2;
  int segment_size = 3;
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0};
  thrust::device_vector<int> d_out(2);

  REQUIRE(cudaSuccess
          == cub::DeviceSegmentedReduce::Reduce(
            d_in.begin(), d_out.begin(), num_segments, segment_size, ::cuda::std::plus<>{}, 0));

  thrust::device_vector<int> expected{21, 8};
  REQUIRE(d_out == expected);
}

TEST_CASE("Device fixed-size segmented sum works with default environment", "[segmented_reduce][device]")
{
  int num_segments = 2;
  int segment_size = 3;
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0};
  thrust::device_vector<int> d_out(2);

  REQUIRE(cudaSuccess == cub::DeviceSegmentedReduce::Sum(d_in.begin(), d_out.begin(), num_segments, segment_size));

  thrust::device_vector<int> expected{21, 8};
  REQUIRE(d_out == expected);
}

TEST_CASE("Device fixed-size segmented min works with default environment", "[segmented_reduce][device]")
{
  int num_segments = 2;
  int segment_size = 3;
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0};
  thrust::device_vector<int> d_out(2);

  REQUIRE(cudaSuccess == cub::DeviceSegmentedReduce::Min(d_in.begin(), d_out.begin(), num_segments, segment_size));

  thrust::device_vector<int> expected{6, 0};
  REQUIRE(d_out == expected);
}

TEST_CASE("Device fixed-size segmented max works with default environment", "[segmented_reduce][device]")
{
  int num_segments = 2;
  int segment_size = 3;
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0};
  thrust::device_vector<int> d_out(2);

  REQUIRE(cudaSuccess == cub::DeviceSegmentedReduce::Max(d_in.begin(), d_out.begin(), num_segments, segment_size));

  thrust::device_vector<int> expected{8, 5};
  REQUIRE(d_out == expected);
}

TEST_CASE("Device fixed-size segmented argmin works with default environment", "[segmented_reduce][device]")
{
  int num_segments = 2;
  int segment_size = 3;
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0};
  thrust::device_vector<cuda::std::pair<int, int>> d_out(2);

  REQUIRE(cudaSuccess == cub::DeviceSegmentedReduce::ArgMin(d_in.begin(), d_out.begin(), num_segments, segment_size));

  thrust::device_vector<cuda::std::pair<int, int>> expected{{1, 6}, {2, 0}};
  REQUIRE(d_out == expected);
}

TEST_CASE("Device fixed-size segmented argmax works with default environment", "[segmented_reduce][device]")
{
  int num_segments = 2;
  int segment_size = 3;
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0};
  thrust::device_vector<cuda::std::pair<int, int>> d_out(2);

  REQUIRE(cudaSuccess == cub::DeviceSegmentedReduce::ArgMax(d_in.begin(), d_out.begin(), num_segments, segment_size));

  thrust::device_vector<cuda::std::pair<int, int>> expected{{0, 8}, {0, 5}};
  REQUIRE(d_out == expected);
}

#endif

C2H_TEST("Device segmented reduce uses environment", "[segmented_reduce][device]")
{
  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(3);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedReduce::Reduce(
      nullptr,
      expected_bytes_allocated,
      d_in.begin(),
      d_out.begin(),
      num_segments,
      d_offsets_it,
      d_offsets_it + 1,
      ::cuda::std::plus<>{},
      0));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_segmented_reduce(
    d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, ::cuda::std::plus<>{}, 0, env);

  thrust::device_vector<int> expected{26, 12, 3};
  REQUIRE(d_out == expected);
}

C2H_TEST("Device segmented sum uses environment", "[segmented_reduce][device]")
{
  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(3);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedReduce::Sum(
      nullptr, expected_bytes_allocated, d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_segmented_reduce_sum(d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, env);

  thrust::device_vector<int> expected{26, 12, 3};
  REQUIRE(d_out == expected);
}

C2H_TEST("Device segmented min uses environment", "[segmented_reduce][device]")
{
  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(3);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedReduce::Min(
      nullptr, expected_bytes_allocated, d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_segmented_reduce_min(d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, env);

  thrust::device_vector<int> expected{5, 0, 1};
  REQUIRE(d_out == expected);
}

C2H_TEST("Device segmented max uses environment", "[segmented_reduce][device]")
{
  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(3);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedReduce::Max(
      nullptr, expected_bytes_allocated, d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_segmented_reduce_max(d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, env);

  thrust::device_vector<int> expected{8, 9, 2};
  REQUIRE(d_out == expected);
}

C2H_TEST("Device segmented argmin uses environment", "[segmented_reduce][device]")
{
  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<cub::KeyValuePair<int, int>> d_out(3);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedReduce::ArgMin(
      nullptr, expected_bytes_allocated, d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_segmented_reduce_argmin(d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, env);

  thrust::device_vector<cub::KeyValuePair<int, int>> expected{{3, 5}, {1, 0}, {0, 1}};
  REQUIRE(d_out == expected);
}

C2H_TEST("Device segmented argmax uses environment", "[segmented_reduce][device]")
{
  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<cub::KeyValuePair<int, int>> d_out(3);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedReduce::ArgMax(
      nullptr, expected_bytes_allocated, d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_segmented_reduce_argmax(d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, env);

  thrust::device_vector<cub::KeyValuePair<int, int>> expected{{0, 8}, {2, 9}, {1, 2}};
  REQUIRE(d_out == expected);
}

template <int BlockThreads>
struct segmented_reduce_tuning
{
  _CCCL_API constexpr auto operator()(::cuda::compute_capability) const
    -> cub::detail::segmented_reduce::segmented_reduce_policy
  {
    auto rp = cub::detail::reduce::agent_reduce_policy{
      BlockThreads, 1, 1, cub::BLOCK_REDUCE_WARP_REDUCTIONS, cub::LOAD_DEFAULT};
    return {rp,
            cub::detail::segmented_reduce::warp_reduce_policy{BlockThreads, 1, 1, 1, cub::LOAD_DEFAULT},
            cub::detail::segmented_reduce::warp_reduce_policy{BlockThreads, 32, 1, 1, cub::LOAD_DEFAULT}};
  }
};

using block_sizes = c2h::type_list<cuda::std::integral_constant<int, 64>, cuda::std::integral_constant<int, 128>>;

#if TEST_LAUNCH != 1

C2H_TEST("DeviceSegmentedReduce::Reduce can be tuned", "[segmented_reduce][device]", block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(3);
  thrust::device_vector<unsigned int> d_block_size(1);

  block_size_extracting_op<::cuda::std::plus<>> reduce_op{thrust::raw_pointer_cast(d_block_size.data())};

  // We are expecting that `unrelated_tuning` is ignored
  auto env = cuda::execution::tune(segmented_reduce_tuning<target_block_size>{});

  device_segmented_reduce(d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, reduce_op, 0, env);

  thrust::device_vector<int> expected{26, 12, 3};
  REQUIRE(d_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

C2H_TEST("Fixed-size DeviceSegmentedReduce::Reduce can be tuned", "[segmented_reduce][device]", block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  int num_segments = 2;
  int segment_size = 3;
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0};
  thrust::device_vector<int> d_out(2);
  thrust::device_vector<unsigned int> d_block_size(1);

  block_size_extracting_op<::cuda::std::plus<>> reduce_op{thrust::raw_pointer_cast(d_block_size.data())};

  auto env = cuda::execution::tune(segmented_reduce_tuning<target_block_size>{});

  device_segmented_reduce(d_in.begin(), d_out.begin(), num_segments, segment_size, reduce_op, 0, env);

  thrust::device_vector<int> expected{21, 8};
  REQUIRE(d_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

C2H_TEST("DeviceSegmentedReduce::Sum can be tuned", "[segmented_reduce][device]", block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_out(3);
  thrust::device_vector<unsigned int> d_block_size(1);

  // use block_size_recording_iterator to embed blockDim info in the input type and query after
  // since Sum can not take a custom reduction_op
  auto d_in = block_size_extracting_constant_iterator(1, thrust::raw_pointer_cast(d_block_size.data()));

  auto env = cuda::execution::tune(segmented_reduce_tuning<target_block_size>{});

  device_segmented_reduce_sum(d_in, d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, env);

  thrust::device_vector<int> expected{4, 3, 2};
  REQUIRE(d_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

C2H_TEST("Fixed-size DeviceSegmentedReduce::Sum can be tuned", "[segmented_reduce][device]", block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  int num_segments = 2;
  int segment_size = 3;
  thrust::device_vector<int> d_out(2);
  thrust::device_vector<unsigned int> d_block_size(1);

  // use block_size_recording_iterator to embed blockDim info in the input type and query after
  // since Sum can not take a custom reduction_op
  auto d_in = block_size_extracting_constant_iterator(1, thrust::raw_pointer_cast(d_block_size.data()));

  auto env = cuda::execution::tune(segmented_reduce_tuning<target_block_size>{});

  device_segmented_reduce_sum(d_in, d_out.begin(), num_segments, segment_size, env);

  thrust::device_vector<int> expected{3, 3};
  REQUIRE(d_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

C2H_TEST("DeviceSegmentedReduce::Min can be tuned", "[segmented_reduce][device]", block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_out(3);
  thrust::device_vector<unsigned int> d_block_size(1);

  auto d_in = block_size_extracting_constant_iterator(1, thrust::raw_pointer_cast(d_block_size.data()));
  auto env  = cuda::execution::tune(segmented_reduce_tuning<target_block_size>{});

  device_segmented_reduce_min(d_in, d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, env);

  thrust::device_vector<int> expected{1, 1, 1};
  REQUIRE(d_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

C2H_TEST("Fixed-size DeviceSegmentedReduce::Min can be tuned", "[segmented_reduce][device]", block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  int num_segments = 2;
  int segment_size = 3;
  thrust::device_vector<int> d_out(2);
  thrust::device_vector<unsigned int> d_block_size(1);

  auto d_in = block_size_extracting_constant_iterator(1, thrust::raw_pointer_cast(d_block_size.data()));
  auto env  = cuda::execution::tune(segmented_reduce_tuning<target_block_size>{});

  device_segmented_reduce_min(d_in, d_out.begin(), num_segments, segment_size, env);

  thrust::device_vector<int> expected{1, 1};
  REQUIRE(d_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

C2H_TEST("DeviceSegmentedReduce::Max can be tuned", "[segmented_reduce][device]", block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_out(3);
  thrust::device_vector<unsigned int> d_block_size(1);

  auto d_in = block_size_extracting_constant_iterator(1, thrust::raw_pointer_cast(d_block_size.data()));
  auto env  = cuda::execution::tune(segmented_reduce_tuning<target_block_size>{});

  device_segmented_reduce_max(d_in, d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, env);

  thrust::device_vector<int> expected{1, 1, 1};
  REQUIRE(d_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

C2H_TEST("Fixed-size DeviceSegmentedReduce::Max can be tuned", "[segmented_reduce][device]", block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  int num_segments = 2;
  int segment_size = 3;
  thrust::device_vector<int> d_out(2);
  thrust::device_vector<unsigned int> d_block_size(1);

  auto d_in = block_size_extracting_constant_iterator(1, thrust::raw_pointer_cast(d_block_size.data()));
  auto env  = cuda::execution::tune(segmented_reduce_tuning<target_block_size>{});

  device_segmented_reduce_max(d_in, d_out.begin(), num_segments, segment_size, env);

  thrust::device_vector<int> expected{1, 1};
  REQUIRE(d_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

C2H_TEST("DeviceSegmentedReduce::ArgMin can be tuned", "[segmented_reduce][device]", block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<cub::KeyValuePair<int, int>> d_out(3);
  thrust::device_vector<unsigned int> d_block_size(1);

  auto d_in = block_size_extracting_constant_iterator(1, thrust::raw_pointer_cast(d_block_size.data()));
  auto env  = cuda::execution::tune(segmented_reduce_tuning<target_block_size>{});

  device_segmented_reduce_argmin(d_in, d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, env);

  thrust::host_vector<cub::KeyValuePair<int, int>> h_out(d_out);
  for (int i = 0; i < num_segments; i++)
  {
    REQUIRE(h_out[i].key == 0);
    REQUIRE(h_out[i].value == 1);
  }
  REQUIRE(d_block_size[0] == target_block_size);
}

C2H_TEST("Fixed-size DeviceSegmentedReduce::ArgMin can be tuned", "[segmented_reduce][device]", block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  int num_segments = 2;
  int segment_size = 3;
  thrust::device_vector<cuda::std::pair<int, int>> d_out(2);
  thrust::device_vector<unsigned int> d_block_size(1);

  auto d_in = block_size_extracting_constant_iterator(1, thrust::raw_pointer_cast(d_block_size.data()));
  auto env  = cuda::execution::tune(segmented_reduce_tuning<target_block_size>{});

  device_segmented_reduce_argmin(d_in, d_out.begin(), num_segments, segment_size, env);

  thrust::device_vector<cuda::std::pair<int, int>> expected{{0, 1}, {0, 1}};
  REQUIRE(d_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

C2H_TEST("DeviceSegmentedReduce::ArgMax can be tuned", "[segmented_reduce][device]", block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<cub::KeyValuePair<int, int>> d_out(3);
  thrust::device_vector<unsigned int> d_block_size(1);

  auto d_in = block_size_extracting_constant_iterator(1, thrust::raw_pointer_cast(d_block_size.data()));
  auto env  = cuda::execution::tune(segmented_reduce_tuning<target_block_size>{});

  device_segmented_reduce_argmax(d_in, d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, env);

  thrust::host_vector<cub::KeyValuePair<int, int>> h_out(d_out);
  for (int i = 0; i < num_segments; i++)
  {
    REQUIRE(h_out[i].key == 0);
    REQUIRE(h_out[i].value == 1);
  }
  REQUIRE(d_block_size[0] == target_block_size);
}

C2H_TEST("Fixed-size DeviceSegmentedReduce::ArgMax can be tuned", "[segmented_reduce][device]", block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  int num_segments = 2;
  int segment_size = 3;
  thrust::device_vector<cuda::std::pair<int, int>> d_out(2);
  thrust::device_vector<unsigned int> d_block_size(1);

  auto d_in = block_size_extracting_constant_iterator(1, thrust::raw_pointer_cast(d_block_size.data()));
  auto env  = cuda::execution::tune(segmented_reduce_tuning<target_block_size>{});

  device_segmented_reduce_argmax(d_in, d_out.begin(), num_segments, segment_size, env);

  thrust::device_vector<cuda::std::pair<int, int>> expected{{0, 1}, {0, 1}};
  REQUIRE(d_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

#endif // TEST_LAUNCH != 1
