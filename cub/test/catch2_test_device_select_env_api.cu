// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "insert_nested_NVTX_range_guard.h"
// above header needs to be included first

#include <cub/device/device_select.cuh>

#include <thrust/device_vector.h>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/require.h>

#include "catch2_test_device_select_common.cuh"
#include <c2h/catch2_test_helper.h>

C2H_TEST("cub::DeviceSelect::If accepts determinism requirements", "[select][env]")
{
  // example-begin select-if-env-determinism
  auto input        = c2h::device_vector<int>{1, 2, 3, 4, 5, 6, 7, 8};
  auto output       = c2h::device_vector<int>(4);
  auto num_selected = c2h::device_vector<int>(1);
  less_than_t<int> le{input[input.size() / 2]};

  auto env = cuda::execution::require(cuda::execution::determinism::run_to_run);

  cub::DeviceSelect::If(input.begin(), output.begin(), num_selected.begin(), input.size(), le, env);

  c2h::device_vector<int> expected_output{1, 2, 3, 4};
  c2h::device_vector<int> expected_num_selected{4};
  // example-end select-if-env-determinism

  REQUIRE(output == expected_output);
  REQUIRE(num_selected == expected_num_selected);
}

C2H_TEST("cub::DeviceSelect::If accepts stream", "[select][env]")
{
  // example-begin flagged-env-determinism
  auto input        = c2h::device_vector<int>{1, 2, 3, 4, 5, 6, 7, 8};
  auto output       = c2h::device_vector<int>(4);
  auto num_selected = c2h::device_vector<int>(4);
  less_than_t<int> le{input[input.size() / 2]};

  cudaStream_t legacy_stream = 0;
  cuda::stream_ref stream_ref{legacy_stream};

  cub::DeviceSelect::If(input.begin(), output.begin(), num_selected.begin(), input.size(), le, stream_ref);

  c2h::device_vector<int> expected_output{1, 2, 3, 4};
  c2h::device_vector<int> expected_num_selected{4, 0, 0, 0};
  // example-end flagged-env-determinism

  REQUIRE(output == expected_output);
  REQUIRE(num_selected == expected_num_selected);
}
