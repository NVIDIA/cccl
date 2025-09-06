// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_select.cuh>

#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>

#include "catch2_test_env_launch_helper.h"

DECLARE_LAUNCH_WRAPPER(cub::DeviceSelect::If, device_select_if);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSelect::FlaggedIf, device_select_flagged_if);

// %PARAM% TEST_LAUNCH lid 0:1:2

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/require.h>

#include <c2h/catch2_test_helper.h>

namespace stdexec = cuda::std::execution;

// Test helper struct
struct less_than_t
{
  int threshold;
  
  __host__ __device__ less_than_t(int t) : threshold(t) {}
  
  __host__ __device__ bool operator()(const int& x) const
  {
    return x < threshold;
  }
};

struct is_even_t
{
  __host__ __device__ bool operator()(const int& flag) const
  {
    return (flag % 2) == 0;
  }
};

// Test DeviceSelect::If with environment
using TestTypes = c2h::type_list<
  std::tuple<cuda::std::execution::env<>, cuda::execution::determinism::run_to_run_t>
>;

TEMPLATE_LIST_TEST_CASE("Device select uses environment", "[select][device]", TestTypes)
{
  using determinism_t = c2h::get<1, TestType>;
  using num_items_t = int;
  
  num_items_t num_items = GENERATE(1 << 4, 1 << 10);
  auto d_in = thrust::device_vector<int>(num_items);
  auto d_num_selected_out = thrust::device_vector<num_items_t>(1);
  
  // Initialize with sequential values: 0, 1, 2, 3, ...
  thrust::sequence(d_in.begin(), d_in.end());
  
  less_than_t select_op(num_items / 2); // Select first half

  if constexpr (std::is_same_v<determinism_t, cuda::execution::determinism::run_to_run_t>)
  {
    // Test with run_to_run determinism
    auto env = cuda::execution::require(determinism_t{});
    
    cub::DeviceSelect::If(d_in.begin(), d_num_selected_out.data(), num_items, select_op, env);
    
    // Should select num_items/2 elements
    REQUIRE(d_num_selected_out[0] == num_items / 2);
    
    // Check that selected elements are correct (first half, compacted)
    d_in.resize(d_num_selected_out[0]);
    for (int i = 0; i < d_num_selected_out[0]; ++i)
    {
      REQUIRE(d_in[i] == i);
    }
  }
}

C2H_TEST("Device select If with environment handles empty input", "[select][device]")
{
  auto d_in = thrust::device_vector<int>{};
  auto d_num_selected_out = thrust::device_vector<int>(1);
  less_than_t select_op(5);
  
  auto env = cuda::execution::require(cuda::execution::determinism::run_to_run);
  
  cub::DeviceSelect::If(d_in.begin(), d_num_selected_out.data(), 0, select_op, env);
  
  REQUIRE(d_num_selected_out[0] == 0);
}

C2H_TEST("Device select FlaggedIf with environment", "[select][device]")
{
  auto d_in = thrust::device_vector<int>{0, 1, 2, 3, 4, 5, 6, 7};
  auto d_flags = thrust::device_vector<int>{8, 6, 7, 5, 3, 0, 9, 3}; // even flags: 8, 6, 0 -> positions 0, 1, 5
  auto d_out = thrust::device_vector<int>(d_in.size());
  auto d_num_selected_out = thrust::device_vector<int>(1);
  is_even_t is_even{};
  
  auto env = cuda::execution::require(cuda::execution::determinism::run_to_run);
  
  cub::DeviceSelect::FlaggedIf(d_in.begin(), d_flags.begin(), d_out.begin(), d_num_selected_out.data(), d_in.size(), is_even, env);
  
  REQUIRE(d_num_selected_out[0] == 3);
  d_out.resize(d_num_selected_out[0]);
  
  // Expected: positions 0, 1, 5 -> values 0, 1, 5
  thrust::device_vector<int> expected{0, 1, 5};
  REQUIRE(d_out == expected);
}

C2H_TEST("Device select FlaggedIf in-place with environment", "[select][device]")
{
  auto d_data = thrust::device_vector<int>{0, 1, 2, 3, 4, 5, 6, 7};
  auto d_flags = thrust::device_vector<int>{8, 6, 7, 5, 3, 0, 9, 3}; // even flags: 8, 6, 0 -> positions 0, 1, 5
  auto d_num_selected_out = thrust::device_vector<int>(1);
  is_even_t is_even{};
  
  auto env = cuda::execution::require(cuda::execution::determinism::run_to_run);
  
  cub::DeviceSelect::FlaggedIf(d_data.begin(), d_flags.begin(), d_num_selected_out.data(), d_data.size(), is_even, env);
  
  REQUIRE(d_num_selected_out[0] == 3);
  d_data.resize(d_num_selected_out[0]);
  
  // Expected: positions 0, 1, 5 -> values 0, 1, 5
  thrust::device_vector<int> expected{0, 1, 5};
  REQUIRE(d_data == expected);
}

C2H_TEST("Device select with stream reference", "[select][device]")
{
  auto d_in = thrust::device_vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  auto d_num_selected_out = thrust::device_vector<int>(1);
  less_than_t select_op(5); // Select values < 5
  
  cudaStream_t stream = 0;
  cuda::stream_ref stream_ref{stream};
  
  cub::DeviceSelect::If(d_in.begin(), d_num_selected_out.data(), d_in.size(), select_op, stream_ref);
  
  REQUIRE(d_num_selected_out[0] == 5);
  d_in.resize(d_num_selected_out[0]);
  
  thrust::device_vector<int> expected{0, 1, 2, 3, 4};
  REQUIRE(d_in == expected);
}