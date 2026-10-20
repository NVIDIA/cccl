// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cuda/devices>
#include <cuda/stream>

#include <c2h/catch2_test_macros.h>
#include <c2h/device_and_stream.h>

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

[[nodiscard]] inline device_and_owning_stream make_current_device_and_owning_stream()
{
  const auto device = ::c2h::current_device();
  return {device, ::cuda::stream{device}};
}

// The default stream handle uses whichever device is current when work is submitted.
[[nodiscard]] inline device_and_default_stream_ref current_device_and_default_stream_ref()
{
  return {::c2h::current_device(), ::cuda::stream_ref{::cudaStream_t{}}};
}
} // namespace cub_test
