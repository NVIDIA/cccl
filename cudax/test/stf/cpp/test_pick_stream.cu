//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

//! \file
//! \brief Test the stream picking functionality using execution place abstraction

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

int main()
{
  // Get the number of available devices
  int device_count;
  cuda_safe_call(cudaGetDeviceCount(&device_count));

  // Create async_resources for stream pool management
  async_resources_handle resources;

  // Get current device for comparison
  int current_device;
  cuda_safe_call(cudaGetDevice(&current_device));

  // Test picking streams from execution places
  exec_place current_dev_place = exec_place::current_device();

  // Test getting decorated stream from current device execution place
  decorated_stream dstream1 = current_dev_place.getStream(resources, true);
  EXPECT(dstream1.stream != nullptr);
  EXPECT(dstream1.dev_id == current_device);
  EXPECT(get_device_from_stream(dstream1.stream) == current_device);

  // Test with different computation vs transfer flag
  decorated_stream dstream2 = current_dev_place.getStream(resources, false);
  EXPECT(dstream2.stream != nullptr);
  EXPECT(dstream2.dev_id == current_device);
  EXPECT(get_device_from_stream(dstream2.stream) == current_device);

  // Test with multiple devices if available
  if (device_count > 1)
  {
    // Test picking streams from different device execution places
    for (int test_device = 0; test_device < ::std::min(device_count, 2); ++test_device)
    {
      exec_place dev_place = exec_place::device(test_device);

      decorated_stream dstream_dev = dev_place.getStream(resources, true);
      EXPECT(dstream_dev.stream != nullptr);
      EXPECT(dstream_dev.dev_id == test_device);
      EXPECT(get_device_from_stream(dstream_dev.stream) == test_device);

      // Test different computation flag
      decorated_stream dstream_transfer = dev_place.getStream(resources, false);
      EXPECT(dstream_transfer.stream != nullptr);
      EXPECT(dstream_transfer.dev_id == test_device);
      EXPECT(get_device_from_stream(dstream_transfer.stream) == test_device);
    }
  }

  // Test context stream picking - contexts now respect execution place abstraction
  {
    context ctx;
    // Context pick_stream now uses default_exec_place().getStream() internally
    cudaStream_t stream3 = ctx.pick_stream();
    EXPECT(stream3 != nullptr);
    EXPECT(get_device_from_stream(stream3) == current_device);
    ctx.finalize();
  }

  // Test with graph context - now also uses execution place abstraction
  {
    graph_ctx gctx;
    // Graph context get_stream() now uses default_exec_place().getStream() internally
    cudaStream_t stream4 = gctx.pick_stream();
    EXPECT(stream4 != nullptr);
    EXPECT(get_device_from_stream(stream4) == current_device);
    gctx.finalize();
  }

  // Test context with execution affinity - demonstrates execution place abstraction
  {
    context ctx;

    // Set affinity to a specific device execution place
    if (device_count > 1)
    {
      exec_place dev1_place = exec_place::device(1);
      ctx.set_affinity({::std::make_shared<exec_place>(dev1_place)});

      // Stream should now come from device 1's pool
      cudaStream_t affinity_stream = ctx.pick_stream();
      EXPECT(affinity_stream != nullptr);
      EXPECT(get_device_from_stream(affinity_stream) == 1);
    }

    ctx.finalize();
  }
}
