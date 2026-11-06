// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cuda/devices>
#include <cuda/stream>

#include <cuda_runtime_api.h>

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

// Returns a reference to the legacy default stream without changing the current device.
// The handle uses whichever device is current when work is submitted; it does not
// retain an association with the device that was current when this function was called.
[[nodiscard]] constexpr cuda::stream_ref default_stream_ref() noexcept
{
  return cuda::stream_ref{cuda::stream_ref::value_type{}};
}
} // namespace c2h
