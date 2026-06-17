// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_copy.cuh>

#include <cuda/std/cstdint>

#include <c2h/catch2_test_helper.h>

// Guard: the legacy memory-size query call with all defaults (no explicit stream)
// must resolve unambiguously to the legacy temp-storage overload when the env
// passthrough overload is also visible. If the env overload's SFINAE is too loose,
// this becomes "ambiguous overload" or silently dispatches to env.

C2H_TEST("DeviceCopy::Batched legacy size-query is unambiguous", "[copy][device]")
{
  // DeviceCopy::Batched takes iterator-of-iterators for input/output ranges.
  int** in                        = nullptr;
  int** out                       = nullptr;
  size_t* sizes                   = nullptr;
  size_t bytes                    = 0;
  ::cuda::std::int64_t num_ranges = 0;

  REQUIRE(cudaSuccess == cub::DeviceCopy::Batched(nullptr, bytes, in, out, sizes, num_ranges));
}

// todo(giannis): extract examples from the docs to literalinclude extracts here

// TODO(giannis): move the DeviceCopy::Copy (mdspan) tests from
// catch2_test_device_copy_mdspan_api.cu here, so all DeviceCopy api tests live together.
