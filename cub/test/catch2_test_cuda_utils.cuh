// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cuda/devices>
#include <cuda/stream>

#include <cuda_runtime_api.h>

#include <c2h/catch2_test_macros.h>

namespace cub_test
{
struct device_and_owning_stream
{
  ::cuda::device_ref device;
  ::cuda::stream stream;
};

struct device_and_default_stream_ref
{
  ::cuda::device_ref device;
  ::cuda::stream_ref stream;
};

[[nodiscard]] inline ::cuda::device_ref current_device()
{
  int device{0};
  REQUIRE(::cudaSuccess == ::cudaGetDevice(&device));
  return ::cuda::device_ref{device};
}

[[nodiscard]] inline device_and_owning_stream make_current_device_and_owning_stream()
{
  const auto device = ::cub_test::current_device();
  return {device, ::cuda::stream{device}};
}

// The default stream handle uses whichever device is current when work is submitted.
[[nodiscard]] inline device_and_default_stream_ref current_device_and_default_stream_ref()
{
  return {::cub_test::current_device(), ::cuda::stream_ref{::cudaStream_t{}}};
}
} // namespace cub_test
