// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_radix_sort.cuh>

#include <thrust/device_vector.h>

#include "catch2_test_env_launch_helper.h"

DECLARE_LAUNCH_WRAPPER(cub::DeviceRadixSort::SortPairs, device_radix_sort_pairs);

// %PARAM% TEST_LAUNCH lid 0:1:2

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/require.h>

#include <c2h/catch2_test_helper.h>

namespace stdexec = cuda::std::execution;

#if TEST_LAUNCH == 0

TEST_CASE("Device radix sort pairs works with default environment", "[radix_sort][device]")
{
  auto keys_in    = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out   = c2h::device_vector<int>(7);
  auto values_in  = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out = c2h::device_vector<int>(7);

  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortPairs(
      keys_in.data().get(),
      keys_out.data().get(),
      values_in.data().get(),
      values_out.data().get(),
      static_cast<int>(keys_in.size()),
      0,
      sizeof(int) * 8));

  c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  c2h::device_vector<int> expected_values{5, 4, 3, 1, 2, 0, 6};

  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

#endif

C2H_TEST("Device radix sort pairs uses environment", "[radix_sort][device]")
{
  auto keys_in    = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out   = c2h::device_vector<int>(7);
  auto values_in  = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out = c2h::device_vector<int>(7);

  size_t expected_bytes_allocated{};
  // calculate expected_bytes_allocated - call CUB API directly, not through wrapper
  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortPairs(
      nullptr,
      expected_bytes_allocated,
      keys_in.data().get(),
      keys_out.data().get(),
      values_in.data().get(),
      values_out.data().get(),
      static_cast<int>(keys_in.size()),
      0,
      sizeof(int) * 8));

  auto env = stdexec::env{cuda::execution::require(cuda::execution::determinism::gpu_to_gpu), // determinism
                          expected_allocation_size(expected_bytes_allocated)}; // temp storage size

  device_radix_sort_pairs(
    keys_in.data().get(),
    keys_out.data().get(),
    values_in.data().get(),
    values_out.data().get(),
    static_cast<int>(keys_in.size()),
    0,
    sizeof(int) * 8,
    env);

  c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  c2h::device_vector<int> expected_values{5, 4, 3, 1, 2, 0, 6};

  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

TEST_CASE("Device radix sort pairs uses custom stream", "[radix_sort][device]")
{
  auto keys_in    = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out   = c2h::device_vector<int>(7);
  auto values_in  = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out = c2h::device_vector<int>(7);

  cudaStream_t custom_stream;
  REQUIRE(cudaSuccess == cudaStreamCreate(&custom_stream));

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortPairs(
      nullptr,
      expected_bytes_allocated,
      keys_in.data().get(),
      keys_out.data().get(),
      values_in.data().get(),
      values_out.data().get(),
      static_cast<int>(keys_in.size()),
      0,
      sizeof(int) * 8));

  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, cuda::stream_ref{custom_stream}};
  auto env         = stdexec::env{stream_prop, expected_allocation_size(expected_bytes_allocated)};

  device_radix_sort_pairs(
    keys_in.data().get(),
    keys_out.data().get(),
    values_in.data().get(),
    values_out.data().get(),
    static_cast<int>(keys_in.size()),
    0,
    sizeof(int) * 8,
    env);

  REQUIRE(cudaSuccess == cudaStreamSynchronize(custom_stream));

  c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  c2h::device_vector<int> expected_values{5, 4, 3, 1, 2, 0, 6};

  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);

  REQUIRE(cudaSuccess == cudaStreamDestroy(custom_stream));
}
