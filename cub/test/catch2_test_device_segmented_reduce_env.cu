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

  thrust::host_vector<cub::KeyValuePair<int, int>> h_out(d_out);
  REQUIRE(h_out[0].key == 3);
  REQUIRE(h_out[0].value == 5);
  REQUIRE(h_out[1].key == 1);
  REQUIRE(h_out[1].value == 0);
  REQUIRE(h_out[2].key == 0);
  REQUIRE(h_out[2].value == 1);
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

  thrust::host_vector<cub::KeyValuePair<int, int>> h_out(d_out);
  REQUIRE(h_out[0].key == 0);
  REQUIRE(h_out[0].value == 8);
  REQUIRE(h_out[1].key == 2);
  REQUIRE(h_out[1].value == 9);
  REQUIRE(h_out[2].key == 1);
  REQUIRE(h_out[2].value == 2);
}

TEST_CASE("Device segmented reduce uses custom stream", "[segmented_reduce][device]")
{
  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(3);

  cudaStream_t custom_stream;
  REQUIRE(cudaSuccess == cudaStreamCreate(&custom_stream));

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

  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, cuda::stream_ref{custom_stream}};
  auto env         = stdexec::env{stream_prop, expected_allocation_size(expected_bytes_allocated)};

  device_segmented_reduce(
    d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, ::cuda::std::plus<>{}, 0, env);

  REQUIRE(cudaSuccess == cudaStreamSynchronize(custom_stream));

  thrust::device_vector<int> expected{26, 12, 3};
  REQUIRE(d_out == expected);

  REQUIRE(cudaSuccess == cudaStreamDestroy(custom_stream));
}

TEST_CASE("Device segmented sum uses custom stream", "[segmented_reduce][device]")
{
  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(3);

  cudaStream_t custom_stream;
  REQUIRE(cudaSuccess == cudaStreamCreate(&custom_stream));

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedReduce::Sum(
      nullptr, expected_bytes_allocated, d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1));

  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, cuda::stream_ref{custom_stream}};
  auto env         = stdexec::env{stream_prop, expected_allocation_size(expected_bytes_allocated)};

  device_segmented_reduce_sum(d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, env);

  REQUIRE(cudaSuccess == cudaStreamSynchronize(custom_stream));

  thrust::device_vector<int> expected{26, 12, 3};
  REQUIRE(d_out == expected);

  REQUIRE(cudaSuccess == cudaStreamDestroy(custom_stream));
}

TEST_CASE("Device segmented min uses custom stream", "[segmented_reduce][device]")
{
  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(3);

  cudaStream_t custom_stream;
  REQUIRE(cudaSuccess == cudaStreamCreate(&custom_stream));

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedReduce::Min(
      nullptr, expected_bytes_allocated, d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1));

  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, cuda::stream_ref{custom_stream}};
  auto env         = stdexec::env{stream_prop, expected_allocation_size(expected_bytes_allocated)};

  device_segmented_reduce_min(d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, env);

  REQUIRE(cudaSuccess == cudaStreamSynchronize(custom_stream));

  thrust::device_vector<int> expected{5, 0, 1};
  REQUIRE(d_out == expected);

  REQUIRE(cudaSuccess == cudaStreamDestroy(custom_stream));
}

TEST_CASE("Device segmented max uses custom stream", "[segmented_reduce][device]")
{
  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(3);

  cudaStream_t custom_stream;
  REQUIRE(cudaSuccess == cudaStreamCreate(&custom_stream));

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedReduce::Max(
      nullptr, expected_bytes_allocated, d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1));

  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, cuda::stream_ref{custom_stream}};
  auto env         = stdexec::env{stream_prop, expected_allocation_size(expected_bytes_allocated)};

  device_segmented_reduce_max(d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, env);

  REQUIRE(cudaSuccess == cudaStreamSynchronize(custom_stream));

  thrust::device_vector<int> expected{8, 9, 2};
  REQUIRE(d_out == expected);

  REQUIRE(cudaSuccess == cudaStreamDestroy(custom_stream));
}

TEST_CASE("Device segmented argmin uses custom stream", "[segmented_reduce][device]")
{
  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<cub::KeyValuePair<int, int>> d_out(3);

  cudaStream_t custom_stream;
  REQUIRE(cudaSuccess == cudaStreamCreate(&custom_stream));

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedReduce::ArgMin(
      nullptr, expected_bytes_allocated, d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1));

  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, cuda::stream_ref{custom_stream}};
  auto env         = stdexec::env{stream_prop, expected_allocation_size(expected_bytes_allocated)};

  device_segmented_reduce_argmin(d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, env);

  REQUIRE(cudaSuccess == cudaStreamSynchronize(custom_stream));

  thrust::host_vector<cub::KeyValuePair<int, int>> h_out(d_out);
  REQUIRE(h_out[0].key == 3);
  REQUIRE(h_out[0].value == 5);
  REQUIRE(h_out[1].key == 1);
  REQUIRE(h_out[1].value == 0);
  REQUIRE(h_out[2].key == 0);
  REQUIRE(h_out[2].value == 1);

  REQUIRE(cudaSuccess == cudaStreamDestroy(custom_stream));
}

TEST_CASE("Device segmented argmax uses custom stream", "[segmented_reduce][device]")
{
  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<cub::KeyValuePair<int, int>> d_out(3);

  cudaStream_t custom_stream;
  REQUIRE(cudaSuccess == cudaStreamCreate(&custom_stream));

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedReduce::ArgMax(
      nullptr, expected_bytes_allocated, d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1));

  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, cuda::stream_ref{custom_stream}};
  auto env         = stdexec::env{stream_prop, expected_allocation_size(expected_bytes_allocated)};

  device_segmented_reduce_argmax(d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, env);

  REQUIRE(cudaSuccess == cudaStreamSynchronize(custom_stream));

  thrust::host_vector<cub::KeyValuePair<int, int>> h_out(d_out);
  REQUIRE(h_out[0].key == 0);
  REQUIRE(h_out[0].value == 8);
  REQUIRE(h_out[1].key == 2);
  REQUIRE(h_out[1].value == 9);
  REQUIRE(h_out[2].key == 1);
  REQUIRE(h_out[2].value == 2);

  REQUIRE(cudaSuccess == cudaStreamDestroy(custom_stream));
}

template <int BlockThreads>
struct reduce_tuning
{
  _CCCL_API constexpr auto operator()(::cuda::arch_id) const -> cub::detail::reduce::reduce_policy
  {
    auto rp = cub::detail::reduce::agent_reduce_policy{
      BlockThreads, 1, 1, cub::BLOCK_REDUCE_WARP_REDUCTIONS, cub::LOAD_DEFAULT};
    return {rp, rp, rp};
  }
};

struct unrelated_policy
{};

struct unrelated_tuning
{
  // should never be called
  auto operator()(cuda::arch_id /*arch*/) const -> unrelated_policy
  {
    throw 1337;
  }
};

using block_sizes = c2h::type_list<cuda::std::integral_constant<int, 32>, cuda::std::integral_constant<int, 64>>;

C2H_TEST("Device segmented sum can be tuned", "[segmented_reduce][device]", block_sizes)
{
  constexpr int target_block_size = c2h::get<0, TestType>::value;

  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 3, 3, 7};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9};
  thrust::device_vector<int> d_out(3);

  // We are expecting that `unrelated_tuning` is ignored
  auto env = cuda::execution::__tune(reduce_tuning<target_block_size>{}, unrelated_tuning{});

  auto error =
    cub::DeviceSegmentedReduce::Sum(d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, env);
  thrust::device_vector<int> expected{21, 0, 17};

  REQUIRE(d_out == expected);
  REQUIRE(error == cudaSuccess);
}
