// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_run_length_encode.cuh>

#include <c2h/catch2_test_helper.h>

// Guard: the legacy memory-size query call with all defaults (no explicit stream)
// must resolve unambiguously to the legacy temp-storage overload when the env
// passthrough overload is also visible. If the env overload's SFINAE is too loose,
// this becomes "ambiguous overload" or silently dispatches to env.

C2H_TEST("DeviceRunLengthEncode::Encode legacy size-query is unambiguous", "[run_length_encode][device]")
{
  int* d_in       = nullptr;
  int* d_unique   = nullptr;
  int* d_lengths  = nullptr;
  int* d_num_runs = nullptr;
  size_t bytes    = 0;
  int n           = 0;

  REQUIRE(cudaSuccess == cub::DeviceRunLengthEncode::Encode(nullptr, bytes, d_in, d_unique, d_lengths, d_num_runs, n));
}

C2H_TEST("DeviceRunLengthEncode::NonTrivialRuns legacy size-query is unambiguous", "[run_length_encode][device]")
{
  int* d_in       = nullptr;
  int* d_offsets  = nullptr;
  int* d_lengths  = nullptr;
  int* d_num_runs = nullptr;
  size_t bytes    = 0;
  int n           = 0;

  REQUIRE(cudaSuccess
          == cub::DeviceRunLengthEncode::NonTrivialRuns(nullptr, bytes, d_in, d_offsets, d_lengths, d_num_runs, n));
}

// todo(giannis): extract examples from the docs to literalinclude extracts here
