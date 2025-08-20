// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "insert_nested_NVTX_range_guard.h"
// above header needs to be included first

#include <cub/device/device_select.cuh>

#include <thrust/device_vector.h>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/require.h>

#include <c2h/catch2_test_helper.h>

// Simple predicate functor for testing
struct is_even_t
{
  __host__ __device__ __forceinline__ bool operator()(const int& a) const
  {
    return (a % 2) == 0;
  }
};

C2H_TEST("cub::DeviceSelect::If accepts determinism requirements", "[select][env]")
{
  // example-begin select-if-env-determinism
  auto data  = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6, 7};
  auto num_selected = c2h::device_vector<int>(1);
  is_even_t is_even{};

  auto env = cuda::execution::require(cuda::execution::determinism::run_to_run);

  cub::DeviceSelect::If(data.begin(), num_selected.data(), data.size(), is_even, env);

  REQUIRE(num_selected[0] == 4); // Expecting {0, 2, 4, 6}
  data.resize(num_selected[0]);
  c2h::device_vector<int> expected{0, 2, 4, 6};
  REQUIRE(data == expected);
  // example-end select-if-env-determinism
}

C2H_TEST("cub::DeviceSelect::If accepts stream", "[select][env]")
{
  // example-begin select-if-env-stream
  auto data  = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6, 7};
  auto num_selected = c2h::device_vector<int>(1);
  is_even_t is_even{};

  cudaStream_t legacy_stream = 0;
  cuda::stream_ref stream_ref{legacy_stream};

  cub::DeviceSelect::If(data.begin(), num_selected.data(), data.size(), is_even, stream_ref);

  REQUIRE(num_selected[0] == 4); // Expecting {0, 2, 4, 6}
  data.resize(num_selected[0]);
  c2h::device_vector<int> expected{0, 2, 4, 6};
  REQUIRE(data == expected);
  // example-end select-if-env-stream
}

C2H_TEST("cub::DeviceSelect::FlaggedIf accepts determinism requirements", "[select][env]")
{
  // example-begin select-flaggedif-env-determinism
  auto input  = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6, 7};
  auto flags  = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9, 3};
  auto output = c2h::device_vector<int>(input.size());
  auto num_selected = c2h::device_vector<int>(1);
  is_even_t is_even{};

  auto env = cuda::execution::require(cuda::execution::determinism::run_to_run);

  cub::DeviceSelect::FlaggedIf(input.begin(), flags.begin(), output.begin(), num_selected.data(), input.size(), is_even, env);

  REQUIRE(num_selected[0] == 3); // Expecting {0, 1, 5}
  output.resize(num_selected[0]);
  c2h::device_vector<int> expected{0, 1, 5};
  REQUIRE(output == expected);
  // example-end select-flaggedif-env-determinism
}

C2H_TEST("cub::DeviceSelect::FlaggedIf accepts stream", "[select][env]")
{
  // example-begin select-flaggedif-env-stream
  auto input  = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6, 7};
  auto flags  = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9, 3};
  auto output = c2h::device_vector<int>(input.size());
  auto num_selected = c2h::device_vector<int>(1);
  is_even_t is_even{};

  cudaStream_t legacy_stream = 0;
  cuda::stream_ref stream_ref{legacy_stream};

  cub::DeviceSelect::FlaggedIf(input.begin(), flags.begin(), output.begin(), num_selected.data(), input.size(), is_even, stream_ref);

  REQUIRE(num_selected[0] == 3); // Expecting {0, 1, 5}
  output.resize(num_selected[0]);
  c2h::device_vector<int> expected{0, 1, 5};
  REQUIRE(output == expected);
  // example-end select-flaggedif-env-stream
}

C2H_TEST("cub::DeviceSelect::FlaggedIf in-place accepts determinism requirements", "[select][env]")
{
  // example-begin select-flaggedif-inplace-env-determinism
  auto data  = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6, 7};
  auto flags = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9, 3};
  auto num_selected = c2h::device_vector<int>(1);
  is_even_t is_even{};

  auto env = cuda::execution::require(cuda::execution::determinism::run_to_run);

  cub::DeviceSelect::FlaggedIf(data.begin(), flags.begin(), num_selected.data(), data.size(), is_even, env);

  REQUIRE(num_selected[0] == 3); // Expecting {0, 1, 5}
  data.resize(num_selected[0]);
  c2h::device_vector<int> expected{0, 1, 5};
  REQUIRE(data == expected);
  // example-end select-flaggedif-inplace-env-determinism
}

C2H_TEST("cub::DeviceSelect::FlaggedIf in-place accepts stream", "[select][env]")
{
  // example-begin select-flaggedif-inplace-env-stream
  auto data  = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6, 7};
  auto flags = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9, 3};
  auto num_selected = c2h::device_vector<int>(1);
  is_even_t is_even{};

  cudaStream_t legacy_stream = 0;
  cuda::stream_ref stream_ref{legacy_stream};

  cub::DeviceSelect::FlaggedIf(data.begin(), flags.begin(), num_selected.data(), data.size(), is_even, stream_ref);

  REQUIRE(num_selected[0] == 3); // Expecting {0, 1, 5}
  data.resize(num_selected[0]);
  c2h::device_vector<int> expected{0, 1, 5};
  REQUIRE(data == expected);
  // example-end select-flaggedif-inplace-env-stream
}