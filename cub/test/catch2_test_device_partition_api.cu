// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_partition.cuh>

#include <c2h/catch2_test_helper.h>

// Guard: the legacy memory-size query call with all defaults (no explicit stream)
// must resolve unambiguously to the legacy temp-storage overload when the env
// passthrough overload is also visible. If the env overload's SFINAE is too loose,
// this becomes "ambiguous overload" or silently dispatches to env.

C2H_TEST("DevicePartition::Flagged legacy size-query is unambiguous", "[partition][device]")
{
  int* d_in           = nullptr;
  int* d_flags        = nullptr;
  int* d_out          = nullptr;
  int* d_num_selected = nullptr;
  size_t bytes        = 0;
  int n               = 0;

  REQUIRE(cudaSuccess == cub::DevicePartition::Flagged(nullptr, bytes, d_in, d_flags, d_out, d_num_selected, n));
}

C2H_TEST("DevicePartition::If legacy size-query is unambiguous", "[partition][device]")
{
  int* d_in           = nullptr;
  int* d_out          = nullptr;
  int* d_num_selected = nullptr;
  size_t bytes        = 0;
  int n               = 0;

  REQUIRE(
    cudaSuccess == cub::DevicePartition::If(nullptr, bytes, d_in, d_out, d_num_selected, n, ::cuda::always_true{}));
}

C2H_TEST("DevicePartition::If three-way legacy size-query is unambiguous", "[partition][device]")
{
  int* d_in           = nullptr;
  int* d_first_part   = nullptr;
  int* d_second_part  = nullptr;
  int* d_unselected   = nullptr;
  int* d_num_selected = nullptr;
  size_t bytes        = 0;
  int n               = 0;

  REQUIRE(
    cudaSuccess
    == cub::DevicePartition::If(
      nullptr,
      bytes,
      d_in,
      d_first_part,
      d_second_part,
      d_unselected,
      d_num_selected,
      n,
      ::cuda::always_true{},
      ::cuda::always_true{}));
}

// todo(giannis): extract examples from the docs to literalinclude extracts here
