// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_segmented_reduce.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

#include <cuda/__functional/call_or.h>
#include <cuda/std/cstdint>

#include <sstream>

#include "block_size_extracting_helpers.h"
#include "catch2_test_custom_streams.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/generators.h>

DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceSegmentedReduce::Reduce, device_segmented_reduce);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceSegmentedReduce::Sum, device_segmented_reduce_sum);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceSegmentedReduce::Min, device_segmented_reduce_min);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceSegmentedReduce::Max, device_segmented_reduce_max);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceSegmentedReduce::ArgMin, device_segmented_reduce_argmin);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceSegmentedReduce::ArgMax, device_segmented_reduce_argmax);

DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedReduce::Reduce, device_segmented_reduce_two_phase);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedReduce::Sum, device_segmented_reduce_sum_two_phase);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedReduce::Min, device_segmented_reduce_min_two_phase);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedReduce::Max, device_segmented_reduce_max_two_phase);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedReduce::ArgMin, device_segmented_reduce_argmin_two_phase);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedReduce::ArgMax, device_segmented_reduce_argmax_two_phase);

// %PARAM% TEST_LAUNCH lid 0:1:2

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/require.h>

#include "cub_test_macros.h"

namespace stdexec = cuda::std::execution;

// Test data: 3 segments: {8,6,7,5, 3,0,9, 1,2} with offsets {0,4,7,9}

#if TEST_LAUNCH == 0

CUB_TEST_CASE("Device segmented reduce works with default environment", "[segmented_reduce][device]", CUB_SMALL)
{
  const int num_segments               = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(3);

  REQUIRE(cudaSuccess
          == cub::DeviceSegmentedReduce::Reduce(
            d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, ::cuda::std::plus<>{}, 0));

  const thrust::device_vector<int> expected{26, 12, 3};
  REQUIRE(d_out == expected);
}

CUB_TEST_CASE("Device segmented sum works with default environment", "[segmented_reduce][device]", CUB_SMALL)
{
  const int num_segments               = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(3);

  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedReduce::Sum(d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1));

  const thrust::device_vector<int> expected{26, 12, 3};
  REQUIRE(d_out == expected);
}

CUB_TEST_CASE("Device segmented min works with default environment", "[segmented_reduce][device]", CUB_SMALL)
{
  const int num_segments               = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(3);

  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedReduce::Min(d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1));

  const thrust::device_vector<int> expected{5, 0, 1};
  REQUIRE(d_out == expected);
}

CUB_TEST_CASE("Device segmented max works with default environment", "[segmented_reduce][device]", CUB_SMALL)
{
  const int num_segments               = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(3);

  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedReduce::Max(d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1));

  const thrust::device_vector<int> expected{8, 9, 2};
  REQUIRE(d_out == expected);
}

CUB_TEST_CASE("Device segmented argmin works with default environment", "[segmented_reduce][device]", CUB_SMALL)
{
  const int num_segments               = 3;
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

CUB_TEST_CASE("Device segmented argmax works with default environment", "[segmented_reduce][device]", CUB_SMALL)
{
  const int num_segments               = 3;
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

CUB_TEST_CASE("Device fixed-size segmented reduce works with default environment",
              "[segmented_reduce][device]",
              CUB_SMALL)
{
  const int num_segments = 2;
  const int segment_size = 3;
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0};
  thrust::device_vector<int> d_out(2);

  REQUIRE(cudaSuccess
          == cub::DeviceSegmentedReduce::Reduce(
            d_in.begin(), d_out.begin(), num_segments, segment_size, ::cuda::std::plus<>{}, 0));

  const thrust::device_vector<int> expected{21, 8};
  REQUIRE(d_out == expected);
}

CUB_TEST_CASE("Device fixed-size segmented sum works with default environment", "[segmented_reduce][device]", CUB_SMALL)
{
  const int num_segments = 2;
  const int segment_size = 3;
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0};
  thrust::device_vector<int> d_out(2);

  REQUIRE(cudaSuccess == cub::DeviceSegmentedReduce::Sum(d_in.begin(), d_out.begin(), num_segments, segment_size));

  const thrust::device_vector<int> expected{21, 8};
  REQUIRE(d_out == expected);
}

CUB_TEST_CASE("Device fixed-size segmented min works with default environment", "[segmented_reduce][device]", CUB_SMALL)
{
  const int num_segments = 2;
  const int segment_size = 3;
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0};
  thrust::device_vector<int> d_out(2);

  REQUIRE(cudaSuccess == cub::DeviceSegmentedReduce::Min(d_in.begin(), d_out.begin(), num_segments, segment_size));

  const thrust::device_vector<int> expected{6, 0};
  REQUIRE(d_out == expected);
}

CUB_TEST_CASE("Device fixed-size segmented max works with default environment", "[segmented_reduce][device]", CUB_SMALL)
{
  const int num_segments = 2;
  const int segment_size = 3;
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0};
  thrust::device_vector<int> d_out(2);

  REQUIRE(cudaSuccess == cub::DeviceSegmentedReduce::Max(d_in.begin(), d_out.begin(), num_segments, segment_size));

  const thrust::device_vector<int> expected{8, 5};
  REQUIRE(d_out == expected);
}

CUB_TEST_CASE("Device fixed-size segmented argmin works with default environment",
              "[segmented_reduce][device]",
              CUB_SMALL)
{
  const int num_segments = 2;
  const int segment_size = 3;
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0};
  thrust::device_vector<cuda::std::pair<int, int>> d_out(2);

  REQUIRE(cudaSuccess == cub::DeviceSegmentedReduce::ArgMin(d_in.begin(), d_out.begin(), num_segments, segment_size));

  const thrust::device_vector<cuda::std::pair<int, int>> expected{{1, 6}, {2, 0}};
  REQUIRE(d_out == expected);
}

CUB_TEST_CASE("Device fixed-size segmented argmax works with default environment",
              "[segmented_reduce][device]",
              CUB_SMALL)
{
  const int num_segments = 2;
  const int segment_size = 3;
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0};
  thrust::device_vector<cuda::std::pair<int, int>> d_out(2);

  REQUIRE(cudaSuccess == cub::DeviceSegmentedReduce::ArgMax(d_in.begin(), d_out.begin(), num_segments, segment_size));

  const thrust::device_vector<cuda::std::pair<int, int>> expected{{0, 8}, {0, 5}};
  REQUIRE(d_out == expected);
}

#endif

CUB_TEST("Device segmented reduce uses environment", "[segmented_reduce][device]", CUB_SMALL)
{
  const int num_segments               = 3;
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

  const thrust::device_vector<int> expected{26, 12, 3};
  REQUIRE(d_out == expected);
}

CUB_TEST("Device segmented sum uses environment", "[segmented_reduce][device]", CUB_SMALL)
{
  const int num_segments               = 3;
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

  const thrust::device_vector<int> expected{26, 12, 3};
  REQUIRE(d_out == expected);
}

CUB_TEST("Device segmented min uses environment", "[segmented_reduce][device]", CUB_SMALL)
{
  const int num_segments               = 3;
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

  const thrust::device_vector<int> expected{5, 0, 1};
  REQUIRE(d_out == expected);
}

CUB_TEST("Device segmented max uses environment", "[segmented_reduce][device]", CUB_SMALL)
{
  const int num_segments               = 3;
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

  const thrust::device_vector<int> expected{8, 9, 2};
  REQUIRE(d_out == expected);
}

CUB_TEST("Device segmented argmin uses environment", "[segmented_reduce][device]", CUB_SMALL)
{
  const int num_segments               = 3;
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

  const thrust::device_vector<cub::KeyValuePair<int, int>> expected{{3, 5}, {1, 0}, {0, 1}};
  REQUIRE(d_out == expected);
}

CUB_TEST("Device segmented argmax uses environment", "[segmented_reduce][device]", CUB_SMALL)
{
  const int num_segments               = 3;
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

  const thrust::device_vector<cub::KeyValuePair<int, int>> expected{{0, 8}, {2, 9}, {1, 2}};
  REQUIRE(d_out == expected);
}

template <int ThreadsPerBlock>
struct segmented_reduce_tuning
{
  _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability) const -> cub::SegmentedReducePolicy
  {
    auto rp = cub::ReducePassPolicy{ThreadsPerBlock, 1, 1, cub::BLOCK_REDUCE_WARP_REDUCTIONS, cub::LOAD_DEFAULT};
    // need the repetition of the return type for GCC9
    return cub::SegmentedReducePolicy{
      rp,
      cub::SegmentedReduceWarpReducePolicy{ThreadsPerBlock, 32, 1, 1, cub::LOAD_DEFAULT},
      cub::SegmentedReduceWarpReducePolicy{ThreadsPerBlock, 1, 1, 1, cub::LOAD_DEFAULT}};
  }
};

using block_sizes = c2h::type_list<cuda::std::integral_constant<int, 64>, cuda::std::integral_constant<int, 128>>;

#if TEST_LAUNCH != 1

CUB_TEST("DeviceSegmentedReduce::Reduce can be tuned", "[segmented_reduce][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  const int num_segments               = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(3);
  thrust::device_vector<unsigned int> d_block_size(1);

  const block_size_extracting_op<::cuda::std::plus<>> reduce_op{thrust::raw_pointer_cast(d_block_size.data())};

  // We are expecting that `unrelated_tuning` is ignored
  auto env = cuda::execution::tune(segmented_reduce_tuning<target_block_size>{});

  device_segmented_reduce(d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, reduce_op, 0, env);

  const thrust::device_vector<int> expected{26, 12, 3};
  REQUIRE(d_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("Fixed-size DeviceSegmentedReduce::Reduce can be tuned", "[segmented_reduce][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  const int num_segments = 2;
  const int segment_size = 3;
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0};
  thrust::device_vector<int> d_out(2);
  thrust::device_vector<unsigned int> d_block_size(1);

  const block_size_extracting_op<::cuda::std::plus<>> reduce_op{thrust::raw_pointer_cast(d_block_size.data())};

  auto env = cuda::execution::tune(segmented_reduce_tuning<target_block_size>{});

  device_segmented_reduce(d_in.begin(), d_out.begin(), num_segments, segment_size, reduce_op, 0, env);

  const thrust::device_vector<int> expected{21, 8};
  REQUIRE(d_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("DeviceSegmentedReduce::Sum can be tuned", "[segmented_reduce][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  const int num_segments               = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_out(3);
  thrust::device_vector<unsigned int> d_block_size(1);

  // use block_size_recording_iterator to embed blockDim info in the input type and query after
  // since Sum can not take a custom reduction_op
  auto d_in = block_size_extracting_constant_iterator(1, thrust::raw_pointer_cast(d_block_size.data()));

  auto env = cuda::execution::tune(segmented_reduce_tuning<target_block_size>{});

  device_segmented_reduce_sum(d_in, d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, env);

  const thrust::device_vector<int> expected{4, 3, 2};
  REQUIRE(d_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("Fixed-size DeviceSegmentedReduce::Sum can be tuned", "[segmented_reduce][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  const int num_segments = 2;
  const int segment_size = 3;
  thrust::device_vector<int> d_out(2);
  thrust::device_vector<unsigned int> d_block_size(1);

  // use block_size_recording_iterator to embed blockDim info in the input type and query after
  // since Sum can not take a custom reduction_op
  auto d_in = block_size_extracting_constant_iterator(1, thrust::raw_pointer_cast(d_block_size.data()));

  auto env = cuda::execution::tune(segmented_reduce_tuning<target_block_size>{});

  device_segmented_reduce_sum(d_in, d_out.begin(), num_segments, segment_size, env);

  const thrust::device_vector<int> expected{3, 3};
  REQUIRE(d_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("DeviceSegmentedReduce::Min can be tuned", "[segmented_reduce][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  const int num_segments               = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_out(3);
  thrust::device_vector<unsigned int> d_block_size(1);

  auto d_in = block_size_extracting_constant_iterator(1, thrust::raw_pointer_cast(d_block_size.data()));
  auto env  = cuda::execution::tune(segmented_reduce_tuning<target_block_size>{});

  device_segmented_reduce_min(d_in, d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, env);

  const thrust::device_vector<int> expected{1, 1, 1};
  REQUIRE(d_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("Fixed-size DeviceSegmentedReduce::Min can be tuned", "[segmented_reduce][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  const int num_segments = 2;
  const int segment_size = 3;
  thrust::device_vector<int> d_out(2);
  thrust::device_vector<unsigned int> d_block_size(1);

  auto d_in = block_size_extracting_constant_iterator(1, thrust::raw_pointer_cast(d_block_size.data()));
  auto env  = cuda::execution::tune(segmented_reduce_tuning<target_block_size>{});

  device_segmented_reduce_min(d_in, d_out.begin(), num_segments, segment_size, env);

  const thrust::device_vector<int> expected{1, 1};
  REQUIRE(d_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("DeviceSegmentedReduce::Max can be tuned", "[segmented_reduce][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  const int num_segments               = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_out(3);
  thrust::device_vector<unsigned int> d_block_size(1);

  auto d_in = block_size_extracting_constant_iterator(1, thrust::raw_pointer_cast(d_block_size.data()));
  auto env  = cuda::execution::tune(segmented_reduce_tuning<target_block_size>{});

  device_segmented_reduce_max(d_in, d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, env);

  const thrust::device_vector<int> expected{1, 1, 1};
  REQUIRE(d_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("Fixed-size DeviceSegmentedReduce::Max can be tuned", "[segmented_reduce][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  const int num_segments = 2;
  const int segment_size = 3;
  thrust::device_vector<int> d_out(2);
  thrust::device_vector<unsigned int> d_block_size(1);

  auto d_in = block_size_extracting_constant_iterator(1, thrust::raw_pointer_cast(d_block_size.data()));
  auto env  = cuda::execution::tune(segmented_reduce_tuning<target_block_size>{});

  device_segmented_reduce_max(d_in, d_out.begin(), num_segments, segment_size, env);

  const thrust::device_vector<int> expected{1, 1};
  REQUIRE(d_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("DeviceSegmentedReduce::ArgMin can be tuned", "[segmented_reduce][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  const int num_segments               = 3;
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

CUB_TEST("Fixed-size DeviceSegmentedReduce::ArgMin can be tuned", "[segmented_reduce][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  const int num_segments = 2;
  const int segment_size = 3;
  thrust::device_vector<cuda::std::pair<int, int>> d_out(2);
  thrust::device_vector<unsigned int> d_block_size(1);

  auto d_in = block_size_extracting_constant_iterator(1, thrust::raw_pointer_cast(d_block_size.data()));
  auto env  = cuda::execution::tune(segmented_reduce_tuning<target_block_size>{});

  device_segmented_reduce_argmin(d_in, d_out.begin(), num_segments, segment_size, env);

  const thrust::device_vector<cuda::std::pair<int, int>> expected{{0, 1}, {0, 1}};
  REQUIRE(d_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("DeviceSegmentedReduce::ArgMax can be tuned", "[segmented_reduce][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  const int num_segments               = 3;
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

CUB_TEST("Fixed-size DeviceSegmentedReduce::ArgMax can be tuned", "[segmented_reduce][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  const int num_segments = 2;
  const int segment_size = 3;
  thrust::device_vector<cuda::std::pair<int, int>> d_out(2);
  thrust::device_vector<unsigned int> d_block_size(1);

  auto d_in = block_size_extracting_constant_iterator(1, thrust::raw_pointer_cast(d_block_size.data()));
  auto env  = cuda::execution::tune(segmented_reduce_tuning<target_block_size>{});

  device_segmented_reduce_argmax(d_in, d_out.begin(), num_segments, segment_size, env);

  const thrust::device_vector<cuda::std::pair<int, int>> expected{{0, 1}, {0, 1}};
  REQUIRE(d_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

#endif // TEST_LAUNCH != 1

#if TEST_LAUNCH == 0

// The launch wrappers copy their arguments, which the non-copyable stream types of test_with_custom_streams do not
// allow, so the two-phase overloads are called directly with custom streams. Test them with host launch only.

// Runs the two-phase overload wrapped by two_phase: queries the temporary storage size, executes with the allocated
// storage, and checks that all kernels are launched on the stream of env.
template <class EnvT, class TwoPhaseFn>
void run_two_phase(const EnvT& env, TwoPhaseFn two_phase)
{
  const stream_scope scope{cuda::__call_or(cuda::get_stream, cuda::stream_ref{cudaStream_t{}}, env).get()};

  size_t temp_storage_bytes = 0;
  REQUIRE(cudaSuccess == two_phase(nullptr, temp_storage_bytes));

  c2h::device_vector<cuda::std::uint8_t> temp_storage(temp_storage_bytes, thrust::no_init);
  REQUIRE(cudaSuccess == two_phase(thrust::raw_pointer_cast(temp_storage.data()), temp_storage_bytes));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

CUB_TEST_CASE("Device segmented reduce two-phase works with custom streams", "[segmented_reduce][device]", CUB_SMALL)
{
  const int num_segments               = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(3);

  test_with_custom_streams([&](const auto& env) {
    run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
      return cub::DeviceSegmentedReduce::Reduce(
        d_temp_storage,
        temp_storage_bytes,
        d_in.begin(),
        d_out.begin(),
        num_segments,
        d_offsets_it,
        d_offsets_it + 1,
        ::cuda::std::plus<>{},
        0,
        env);
    });

    const thrust::device_vector<int> expected{26, 12, 3};
    REQUIRE(d_out == expected);
  });
}

CUB_TEST_CASE("Device fixed-size segmented reduce two-phase works with custom streams",
              "[segmented_reduce][device]",
              CUB_SMALL)
{
  const int num_segments = 2;
  const int segment_size = 3;
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0};
  thrust::device_vector<int> d_out(2);

  test_with_custom_streams([&](const auto& env) {
    run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
      return cub::DeviceSegmentedReduce::Reduce(
        d_temp_storage,
        temp_storage_bytes,
        d_in.begin(),
        d_out.begin(),
        num_segments,
        segment_size,
        ::cuda::std::plus<>{},
        0,
        env);
    });

    const thrust::device_vector<int> expected{21, 8};
    REQUIRE(d_out == expected);
  });
}

CUB_TEST_CASE("Device segmented sum two-phase works with custom streams", "[segmented_reduce][device]", CUB_SMALL)
{
  const int num_segments               = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(3);

  test_with_custom_streams([&](const auto& env) {
    run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
      return cub::DeviceSegmentedReduce::Sum(
        d_temp_storage,
        temp_storage_bytes,
        d_in.begin(),
        d_out.begin(),
        num_segments,
        d_offsets_it,
        d_offsets_it + 1,
        env);
    });

    const thrust::device_vector<int> expected{26, 12, 3};
    REQUIRE(d_out == expected);
  });
}

CUB_TEST_CASE("Device fixed-size segmented sum two-phase works with custom streams",
              "[segmented_reduce][device]",
              CUB_SMALL)
{
  const int num_segments = 2;
  const int segment_size = 3;
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0};
  thrust::device_vector<int> d_out(2);

  test_with_custom_streams([&](const auto& env) {
    run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
      return cub::DeviceSegmentedReduce::Sum(
        d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), num_segments, segment_size, env);
    });

    const thrust::device_vector<int> expected{21, 8};
    REQUIRE(d_out == expected);
  });
}

CUB_TEST_CASE("Device segmented min two-phase works with custom streams", "[segmented_reduce][device]", CUB_SMALL)
{
  const int num_segments               = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(3);

  test_with_custom_streams([&](const auto& env) {
    run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
      return cub::DeviceSegmentedReduce::Min(
        d_temp_storage,
        temp_storage_bytes,
        d_in.begin(),
        d_out.begin(),
        num_segments,
        d_offsets_it,
        d_offsets_it + 1,
        env);
    });

    const thrust::device_vector<int> expected{5, 0, 1};
    REQUIRE(d_out == expected);
  });
}

CUB_TEST_CASE("Device fixed-size segmented min two-phase works with custom streams",
              "[segmented_reduce][device]",
              CUB_SMALL)
{
  const int num_segments = 2;
  const int segment_size = 3;
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0};
  thrust::device_vector<int> d_out(2);

  test_with_custom_streams([&](const auto& env) {
    run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
      return cub::DeviceSegmentedReduce::Min(
        d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), num_segments, segment_size, env);
    });

    const thrust::device_vector<int> expected{6, 0};
    REQUIRE(d_out == expected);
  });
}

CUB_TEST_CASE("Device segmented max two-phase works with custom streams", "[segmented_reduce][device]", CUB_SMALL)
{
  const int num_segments               = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(3);

  test_with_custom_streams([&](const auto& env) {
    run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
      return cub::DeviceSegmentedReduce::Max(
        d_temp_storage,
        temp_storage_bytes,
        d_in.begin(),
        d_out.begin(),
        num_segments,
        d_offsets_it,
        d_offsets_it + 1,
        env);
    });

    const thrust::device_vector<int> expected{8, 9, 2};
    REQUIRE(d_out == expected);
  });
}

CUB_TEST_CASE("Device fixed-size segmented max two-phase works with custom streams",
              "[segmented_reduce][device]",
              CUB_SMALL)
{
  const int num_segments = 2;
  const int segment_size = 3;
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0};
  thrust::device_vector<int> d_out(2);

  test_with_custom_streams([&](const auto& env) {
    run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
      return cub::DeviceSegmentedReduce::Max(
        d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), num_segments, segment_size, env);
    });

    const thrust::device_vector<int> expected{8, 5};
    REQUIRE(d_out == expected);
  });
}

CUB_TEST_CASE("Device segmented argmin two-phase works with custom streams", "[segmented_reduce][device]", CUB_SMALL)
{
  const int num_segments               = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<cub::KeyValuePair<int, int>> d_out(3);

  test_with_custom_streams([&](const auto& env) {
    run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
      return cub::DeviceSegmentedReduce::ArgMin(
        d_temp_storage,
        temp_storage_bytes,
        d_in.begin(),
        d_out.begin(),
        num_segments,
        d_offsets_it,
        d_offsets_it + 1,
        env);
    });

    const thrust::device_vector<cub::KeyValuePair<int, int>> expected{{3, 5}, {1, 0}, {0, 1}};
    REQUIRE(d_out == expected);
  });
}

CUB_TEST_CASE("Device fixed-size segmented argmin two-phase works with custom streams",
              "[segmented_reduce][device]",
              CUB_SMALL)
{
  const int num_segments = 2;
  const int segment_size = 3;
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0};
  thrust::device_vector<cuda::std::pair<int, int>> d_out(2);

  test_with_custom_streams([&](const auto& env) {
    run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
      return cub::DeviceSegmentedReduce::ArgMin(
        d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), num_segments, segment_size, env);
    });

    const thrust::device_vector<cuda::std::pair<int, int>> expected{{1, 6}, {2, 0}};
    REQUIRE(d_out == expected);
  });
}

CUB_TEST_CASE("Device segmented argmax two-phase works with custom streams", "[segmented_reduce][device]", CUB_SMALL)
{
  const int num_segments               = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<cub::KeyValuePair<int, int>> d_out(3);

  test_with_custom_streams([&](const auto& env) {
    run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
      return cub::DeviceSegmentedReduce::ArgMax(
        d_temp_storage,
        temp_storage_bytes,
        d_in.begin(),
        d_out.begin(),
        num_segments,
        d_offsets_it,
        d_offsets_it + 1,
        env);
    });

    const thrust::device_vector<cub::KeyValuePair<int, int>> expected{{0, 8}, {2, 9}, {1, 2}};
    REQUIRE(d_out == expected);
  });
}

CUB_TEST_CASE("Device fixed-size segmented argmax two-phase works with custom streams",
              "[segmented_reduce][device]",
              CUB_SMALL)
{
  const int num_segments = 2;
  const int segment_size = 3;
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0};
  thrust::device_vector<cuda::std::pair<int, int>> d_out(2);

  test_with_custom_streams([&](const auto& env) {
    run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
      return cub::DeviceSegmentedReduce::ArgMax(
        d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), num_segments, segment_size, env);
    });

    const thrust::device_vector<cuda::std::pair<int, int>> expected{{0, 8}, {0, 5}};
    REQUIRE(d_out == expected);
  });
}

#endif // TEST_LAUNCH == 0

// The graph launch (TEST_LAUNCH == 2) appends its own stream argument, so it cannot be combined with an environment.
#if TEST_LAUNCH != 2

// Runs every two-phase overload with the given legacy stream argument
template <class StreamT>
void check_two_phase_overloads_with_legacy_stream(StreamT stream)
{
  // variable-size segments: {8,6,7,5, 3,0,9, 1,2}
  {
    const int num_segments               = 3;
    thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
    auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
    thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
    thrust::device_vector<int> d_reduce_out(3), d_sum_out(3), d_min_out(3), d_max_out(3);
    thrust::device_vector<cub::KeyValuePair<int, int>> d_argmin_out(3), d_argmax_out(3);

    {
      const stream_scope scope{cudaStream_t{}};
      device_segmented_reduce_two_phase(
        d_in.begin(),
        d_reduce_out.begin(),
        num_segments,
        d_offsets_it,
        d_offsets_it + 1,
        ::cuda::std::plus<>{},
        0,
        stream);
      device_segmented_reduce_sum_two_phase(
        d_in.begin(), d_sum_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, stream);
      device_segmented_reduce_min_two_phase(
        d_in.begin(), d_min_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, stream);
      device_segmented_reduce_max_two_phase(
        d_in.begin(), d_max_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, stream);
      device_segmented_reduce_argmin_two_phase(
        d_in.begin(), d_argmin_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, stream);
      device_segmented_reduce_argmax_two_phase(
        d_in.begin(), d_argmax_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, stream);
    }

    REQUIRE(d_reduce_out == thrust::device_vector<int>{26, 12, 3});
    REQUIRE(d_sum_out == thrust::device_vector<int>{26, 12, 3});
    REQUIRE(d_min_out == thrust::device_vector<int>{5, 0, 1});
    REQUIRE(d_max_out == thrust::device_vector<int>{8, 9, 2});
    REQUIRE(d_argmin_out == thrust::device_vector<cub::KeyValuePair<int, int>>{{3, 5}, {1, 0}, {0, 1}});
    REQUIRE(d_argmax_out == thrust::device_vector<cub::KeyValuePair<int, int>>{{0, 8}, {2, 9}, {1, 2}});
  }

  // fixed-size segments: {8,6,7, 5,3,0}
  {
    const int num_segments = 2;
    const int segment_size = 3;
    thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0};
    thrust::device_vector<int> d_reduce_out(2), d_sum_out(2), d_min_out(2), d_max_out(2);
    thrust::device_vector<cuda::std::pair<int, int>> d_argmin_out(2), d_argmax_out(2);

    {
      const stream_scope scope{cudaStream_t{}};
      device_segmented_reduce_two_phase(
        d_in.begin(), d_reduce_out.begin(), num_segments, segment_size, ::cuda::std::plus<>{}, 0, stream);
      device_segmented_reduce_sum_two_phase(d_in.begin(), d_sum_out.begin(), num_segments, segment_size, stream);
      device_segmented_reduce_min_two_phase(d_in.begin(), d_min_out.begin(), num_segments, segment_size, stream);
      device_segmented_reduce_max_two_phase(d_in.begin(), d_max_out.begin(), num_segments, segment_size, stream);
      device_segmented_reduce_argmin_two_phase(d_in.begin(), d_argmin_out.begin(), num_segments, segment_size, stream);
      device_segmented_reduce_argmax_two_phase(d_in.begin(), d_argmax_out.begin(), num_segments, segment_size, stream);
    }

    REQUIRE(d_reduce_out == thrust::device_vector<int>{21, 8});
    REQUIRE(d_sum_out == thrust::device_vector<int>{21, 8});
    REQUIRE(d_min_out == thrust::device_vector<int>{6, 0});
    REQUIRE(d_max_out == thrust::device_vector<int>{8, 5});
    REQUIRE(d_argmin_out == thrust::device_vector<cuda::std::pair<int, int>>{{1, 6}, {2, 0}});
    REQUIRE(d_argmax_out == thrust::device_vector<cuda::std::pair<int, int>>{{0, 8}, {0, 5}});
  }
}

// Before the environment parameter, the two-phase overloads took `cudaStream_t stream = nullptr`, so callers passing
// nullptr, a literal 0 or a null cudaStream_t for the stream exist. They must keep compiling and keep running on the
// default stream.
CUB_TEST_CASE("Device segmented reduce two-phase overloads accept legacy null stream arguments",
              "[segmented_reduce][device]",
              CUB_SMALL)
{
  SECTION("nullptr")
  {
    check_two_phase_overloads_with_legacy_stream(nullptr);
  }

  SECTION("literal 0")
  {
    check_two_phase_overloads_with_legacy_stream(0);
  }

  SECTION("null cudaStream_t")
  {
    check_two_phase_overloads_with_legacy_stream(cudaStream_t{});
  }
}

CUB_TEST("DeviceSegmentedReduce::Reduce two-phase can be tuned", "[segmented_reduce][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  const int num_segments               = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(3);
  thrust::device_vector<unsigned int> d_block_size(1);

  const block_size_extracting_op<::cuda::std::plus<>> reduce_op{thrust::raw_pointer_cast(d_block_size.data())};

  auto env = cuda::execution::tune(segmented_reduce_tuning<target_block_size>{});

  device_segmented_reduce_two_phase(
    d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, reduce_op, 0, env);

  const thrust::device_vector<int> expected{26, 12, 3};
  REQUIRE(d_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("Fixed-size DeviceSegmentedReduce::Reduce two-phase can be tuned",
         "[segmented_reduce][device]",
         CUB_SMALL,
         block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  const int num_segments = 2;
  const int segment_size = 3;
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0};
  thrust::device_vector<int> d_out(2);
  thrust::device_vector<unsigned int> d_block_size(1);

  const block_size_extracting_op<::cuda::std::plus<>> reduce_op{thrust::raw_pointer_cast(d_block_size.data())};

  auto env = cuda::execution::tune(segmented_reduce_tuning<target_block_size>{});

  device_segmented_reduce_two_phase(d_in.begin(), d_out.begin(), num_segments, segment_size, reduce_op, 0, env);

  const thrust::device_vector<int> expected{21, 8};
  REQUIRE(d_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("DeviceSegmentedReduce::Sum two-phase can be tuned", "[segmented_reduce][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  const int num_segments               = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_out(3);
  thrust::device_vector<unsigned int> d_block_size(1);

  // Sum can not take a custom reduction_op, so the input iterator records the block size
  auto d_in = block_size_extracting_constant_iterator(1, thrust::raw_pointer_cast(d_block_size.data()));
  auto env  = cuda::execution::tune(segmented_reduce_tuning<target_block_size>{});

  device_segmented_reduce_sum_two_phase(d_in, d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, env);

  const thrust::device_vector<int> expected{4, 3, 2};
  REQUIRE(d_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST(
  "Fixed-size DeviceSegmentedReduce::Sum two-phase can be tuned", "[segmented_reduce][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  const int num_segments = 2;
  const int segment_size = 3;
  thrust::device_vector<int> d_out(2);
  thrust::device_vector<unsigned int> d_block_size(1);

  // Sum can not take a custom reduction_op, so the input iterator records the block size
  auto d_in = block_size_extracting_constant_iterator(1, thrust::raw_pointer_cast(d_block_size.data()));
  auto env  = cuda::execution::tune(segmented_reduce_tuning<target_block_size>{});

  device_segmented_reduce_sum_two_phase(d_in, d_out.begin(), num_segments, segment_size, env);

  const thrust::device_vector<int> expected{3, 3};
  REQUIRE(d_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("DeviceSegmentedReduce::Min two-phase can be tuned", "[segmented_reduce][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  const int num_segments               = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_out(3);
  thrust::device_vector<unsigned int> d_block_size(1);

  auto d_in = block_size_extracting_constant_iterator(1, thrust::raw_pointer_cast(d_block_size.data()));
  auto env  = cuda::execution::tune(segmented_reduce_tuning<target_block_size>{});

  device_segmented_reduce_min_two_phase(d_in, d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, env);

  const thrust::device_vector<int> expected{1, 1, 1};
  REQUIRE(d_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST(
  "Fixed-size DeviceSegmentedReduce::Min two-phase can be tuned", "[segmented_reduce][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  const int num_segments = 2;
  const int segment_size = 3;
  thrust::device_vector<int> d_out(2);
  thrust::device_vector<unsigned int> d_block_size(1);

  auto d_in = block_size_extracting_constant_iterator(1, thrust::raw_pointer_cast(d_block_size.data()));
  auto env  = cuda::execution::tune(segmented_reduce_tuning<target_block_size>{});

  device_segmented_reduce_min_two_phase(d_in, d_out.begin(), num_segments, segment_size, env);

  const thrust::device_vector<int> expected{1, 1};
  REQUIRE(d_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("DeviceSegmentedReduce::Max two-phase can be tuned", "[segmented_reduce][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  const int num_segments               = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_out(3);
  thrust::device_vector<unsigned int> d_block_size(1);

  auto d_in = block_size_extracting_constant_iterator(1, thrust::raw_pointer_cast(d_block_size.data()));
  auto env  = cuda::execution::tune(segmented_reduce_tuning<target_block_size>{});

  device_segmented_reduce_max_two_phase(d_in, d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, env);

  const thrust::device_vector<int> expected{1, 1, 1};
  REQUIRE(d_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST(
  "Fixed-size DeviceSegmentedReduce::Max two-phase can be tuned", "[segmented_reduce][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  const int num_segments = 2;
  const int segment_size = 3;
  thrust::device_vector<int> d_out(2);
  thrust::device_vector<unsigned int> d_block_size(1);

  auto d_in = block_size_extracting_constant_iterator(1, thrust::raw_pointer_cast(d_block_size.data()));
  auto env  = cuda::execution::tune(segmented_reduce_tuning<target_block_size>{});

  device_segmented_reduce_max_two_phase(d_in, d_out.begin(), num_segments, segment_size, env);

  const thrust::device_vector<int> expected{1, 1};
  REQUIRE(d_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("DeviceSegmentedReduce::ArgMin two-phase can be tuned", "[segmented_reduce][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  const int num_segments               = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<cub::KeyValuePair<int, int>> d_out(3);
  thrust::device_vector<unsigned int> d_block_size(1);

  auto d_in = block_size_extracting_constant_iterator(1, thrust::raw_pointer_cast(d_block_size.data()));
  auto env  = cuda::execution::tune(segmented_reduce_tuning<target_block_size>{});

  device_segmented_reduce_argmin_two_phase(d_in, d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, env);

  const thrust::device_vector<cub::KeyValuePair<int, int>> expected{{0, 1}, {0, 1}, {0, 1}};
  REQUIRE(d_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("Fixed-size DeviceSegmentedReduce::ArgMin two-phase can be tuned",
         "[segmented_reduce][device]",
         CUB_SMALL,
         block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  const int num_segments = 2;
  const int segment_size = 3;
  thrust::device_vector<cuda::std::pair<int, int>> d_out(2);
  thrust::device_vector<unsigned int> d_block_size(1);

  auto d_in = block_size_extracting_constant_iterator(1, thrust::raw_pointer_cast(d_block_size.data()));
  auto env  = cuda::execution::tune(segmented_reduce_tuning<target_block_size>{});

  device_segmented_reduce_argmin_two_phase(d_in, d_out.begin(), num_segments, segment_size, env);

  const thrust::device_vector<cuda::std::pair<int, int>> expected{{0, 1}, {0, 1}};
  REQUIRE(d_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("DeviceSegmentedReduce::ArgMax two-phase can be tuned", "[segmented_reduce][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  const int num_segments               = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<cub::KeyValuePair<int, int>> d_out(3);
  thrust::device_vector<unsigned int> d_block_size(1);

  auto d_in = block_size_extracting_constant_iterator(1, thrust::raw_pointer_cast(d_block_size.data()));
  auto env  = cuda::execution::tune(segmented_reduce_tuning<target_block_size>{});

  device_segmented_reduce_argmax_two_phase(d_in, d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, env);

  const thrust::device_vector<cub::KeyValuePair<int, int>> expected{{0, 1}, {0, 1}, {0, 1}};
  REQUIRE(d_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("Fixed-size DeviceSegmentedReduce::ArgMax two-phase can be tuned",
         "[segmented_reduce][device]",
         CUB_SMALL,
         block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  const int num_segments = 2;
  const int segment_size = 3;
  thrust::device_vector<cuda::std::pair<int, int>> d_out(2);
  thrust::device_vector<unsigned int> d_block_size(1);

  auto d_in = block_size_extracting_constant_iterator(1, thrust::raw_pointer_cast(d_block_size.data()));
  auto env  = cuda::execution::tune(segmented_reduce_tuning<target_block_size>{});

  device_segmented_reduce_argmax_two_phase(d_in, d_out.begin(), num_segments, segment_size, env);

  const thrust::device_vector<cuda::std::pair<int, int>> expected{{0, 1}, {0, 1}};
  REQUIRE(d_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

// The two-phase overloads read the determinism requirement from the environment like the single-phase ones: both
// supported levels are accepted, and a run_to_run floating-point sum produces the same result as the single-phase
// overload.
CUB_TEST_CASE("Device segmented reduce two-phase honors determinism requirements",
              "[segmented_reduce][device]",
              CUB_SMALL)
{
  const int num_segments = 10;
  const int segment_size = 1'000;
  const int num_items    = num_segments * segment_size;
  c2h::device_vector<float> d_in(num_items, thrust::no_init);
  c2h::gen(C2H_SEED(1), d_in, -100.0f, 100.0f);
  c2h::device_vector<int> d_offsets(num_segments + 1, thrust::no_init);
  thrust::sequence(d_offsets.begin(), d_offsets.end(), 0, segment_size);
  auto d_offsets_it = thrust::raw_pointer_cast(d_offsets.data());

  const auto op = ::cuda::std::plus<float>{};

  c2h::device_vector<float> d_reference(num_segments, thrust::no_init);
  c2h::device_vector<float> d_out(num_segments, thrust::no_init);

  SECTION("run_to_run")
  {
    const auto env = cuda::execution::require(cuda::execution::determinism::run_to_run);

    REQUIRE(cudaSuccess
            == cub::DeviceSegmentedReduce::Reduce(
              d_in.begin(), d_reference.begin(), num_segments, d_offsets_it, d_offsets_it + 1, op, 0.0f, env));
    device_segmented_reduce_two_phase(
      d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, op, 0.0f, env);
    REQUIRE(d_out == d_reference);

    REQUIRE(cudaSuccess
            == cub::DeviceSegmentedReduce::Reduce(
              d_in.begin(), d_reference.begin(), num_segments, segment_size, op, 0.0f, env));
    device_segmented_reduce_two_phase(d_in.begin(), d_out.begin(), num_segments, segment_size, op, 0.0f, env);
    REQUIRE(d_out == d_reference);
  }

  SECTION("not_guaranteed")
  {
    const auto env = cuda::execution::require(cuda::execution::determinism::not_guaranteed);

    // integer sums are exact regardless of the reduction order
    c2h::device_vector<int> d_int_in(num_items, 1);
    c2h::device_vector<int> d_int_out(num_segments, thrust::no_init);
    const c2h::device_vector<int> expected(num_segments, segment_size);

    device_segmented_reduce_sum_two_phase(
      d_int_in.begin(), d_int_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, env);
    REQUIRE(d_int_out == expected);

    device_segmented_reduce_sum_two_phase(d_int_in.begin(), d_int_out.begin(), num_segments, segment_size, env);
    REQUIRE(d_int_out == expected);
  }
}

#endif // TEST_LAUNCH != 2

#if _CCCL_COMPILER(GCC, >=, 8) // gcc 7 cannot preserve constexpr-ness from p1 to p2
CUB_TEST("Test SegmentedReducePolicy properties", "[segmented_reduce][device]", CUB_SMALL)
{
  STATIC_REQUIRE(::cuda::std::semiregular<cub::SegmentedReducePolicy>);
  STATIC_REQUIRE(::cuda::std::is_aggregate_v<cub::SegmentedReducePolicy>);

  STATIC_REQUIRE(::cuda::std::semiregular<cub::SegmentedReduceWarpReducePolicy>);
  STATIC_REQUIRE(::cuda::std::is_aggregate_v<cub::SegmentedReduceWarpReducePolicy>);

  // aggregate init
  constexpr auto p1_large  = cub::ReducePassPolicy{256, 16, 4, cub::BLOCK_REDUCE_WARP_REDUCTIONS, cub::LOAD_LDG};
  constexpr auto p1_medium = cub::SegmentedReduceWarpReducePolicy{256, 32, 16, 4, cub::LOAD_LDG};
  constexpr auto p1_small  = cub::SegmentedReduceWarpReducePolicy{256, 1, 16, 4, cub::LOAD_LDG};
  constexpr auto p1        = cub::SegmentedReducePolicy{p1_large, p1_medium, p1_small};

#  if _CCCL_STD_VER >= 2020
  // designated init
  constexpr auto p2_large = cub::ReducePassPolicy{
    .threads_per_block = 256,
    .items_per_thread  = 16,
    .vec_size          = 4,
    .reduce_algorithm  = cub::BLOCK_REDUCE_WARP_REDUCTIONS,
    .load_modifier     = cub::LOAD_LDG};
  constexpr auto p2_medium = cub::SegmentedReduceWarpReducePolicy{
    .threads_per_block = 256,
    .threads_per_warp  = 32,
    .items_per_thread  = 16,
    .vec_size          = 4,
    .load_modifier     = cub::LOAD_LDG};
  constexpr auto p2_small = cub::SegmentedReduceWarpReducePolicy{
    .threads_per_block = 256,
    .threads_per_warp  = 1,
    .items_per_thread  = 16,
    .vec_size          = 4,
    .load_modifier     = cub::LOAD_LDG};
  constexpr auto p2 =
    cub::SegmentedReducePolicy{.large_reduce = p2_large, .medium_reduce = p2_medium, .small_reduce = p2_small};
#  else // _CCCL_STD_VER >= 2020
  constexpr auto p2_large  = p1_large;
  constexpr auto p2_medium = p1_medium;
  constexpr auto p2_small  = p1_small;
  constexpr auto p2        = p1;
#  endif // _CCCL_STD_VER >= 2020

  // comparison
  STATIC_REQUIRE(p1_large == p2_large);
  STATIC_REQUIRE_FALSE(p1_large != p2_large);

  STATIC_REQUIRE(p1_medium == p2_medium);
  STATIC_REQUIRE_FALSE(p1_medium != p2_medium);

  STATIC_REQUIRE(p1_small == p2_small);
  STATIC_REQUIRE_FALSE(p1_small != p2_small);

  STATIC_REQUIRE(p1 == p2);
  STATIC_REQUIRE_FALSE(p1 != p2);

  auto to_string = [](const auto& p) {
    std::ostringstream os;
    os << p;
    return os.str();
  };
  REQUIRE(to_string(p1_large)
          == "ReducePassPolicy { .threads_per_block = 256, .items_per_thread = 16, .vec_size = 4"
             ", .reduce_algorithm = BLOCK_REDUCE_WARP_REDUCTIONS, .load_modifier = LOAD_LDG }");
  REQUIRE(to_string(p1_medium)
          == "SegmentedReduceWarpReducePolicy { .threads_per_block = 256, .threads_per_warp = 32"
             ", .items_per_thread = 16, .vec_size = 4, .load_modifier = LOAD_LDG }");
  REQUIRE(to_string(p1_small)
          == "SegmentedReduceWarpReducePolicy { .threads_per_block = 256, .threads_per_warp = 1"
             ", .items_per_thread = 16, .vec_size = 4, .load_modifier = LOAD_LDG }");
  REQUIRE(
    to_string(p1)
    == "SegmentedReducePolicy { .large_reduce = ReducePassPolicy { .threads_per_block = 256"
       ", .items_per_thread = 16, .vec_size = 4"
       ", .reduce_algorithm = BLOCK_REDUCE_WARP_REDUCTIONS, .load_modifier = LOAD_LDG }"
       ", .medium_reduce = SegmentedReduceWarpReducePolicy { .threads_per_block = 256"
       ", .threads_per_warp = 32, .items_per_thread = 16, .vec_size = 4, .load_modifier = LOAD_LDG }"
       ", .small_reduce = SegmentedReduceWarpReducePolicy { .threads_per_block = 256"
       ", .threads_per_warp = 1, .items_per_thread = 16, .vec_size = 4, .load_modifier = LOAD_LDG } }");
}
#endif // _CCCL_COMPILER(GCC, >=, 8)
