// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cuda/devices>
#include <cuda/stream>

#include <catch2/catch_test_macros.hpp>

namespace c2h
{
[[nodiscard]] inline cuda::device_ref current_device()
{
  int device_id{};
  REQUIRE(cudaSuccess == cudaGetDevice(&device_id));
  return cuda::devices[device_id];
}

[[nodiscard]] inline cuda::stream make_current_device_stream()
{
  return cuda::stream{current_device()};
}
} // namespace c2h
