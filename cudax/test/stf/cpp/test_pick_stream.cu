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

  // Create async_resources_handle for stream pool management.
  // This can be used independently of any CUDASTF context.
  async_resources_handle resources;

  // Get current device for comparison
  int current_device;
  cuda_safe_call(cudaGetDevice(&current_device));

  // ==========================================================================
  // Test exec_place::pick_stream() - returns cudaStream_t directly
  // ==========================================================================
  {
    exec_place place = exec_place::current_device();

    // pick_stream() returns a cudaStream_t directly (simpler API)
    cudaStream_t stream = place.pick_stream(resources);
    EXPECT(stream != nullptr);
    EXPECT(get_device_from_stream(stream) == current_device);

    // The for_computation parameter is a performance hint (defaults to true).
    // When true, uses the computation stream pool; when false, uses the
    // transfer stream pool. Using separate pools can improve overlapping.
    cudaStream_t compute_stream  = place.pick_stream(resources, true);
    cudaStream_t transfer_stream = place.pick_stream(resources, false);
    EXPECT(compute_stream != nullptr);
    EXPECT(transfer_stream != nullptr);
  }

  // ==========================================================================
  // Test exec_place::getStream() - returns decorated_stream with metadata
  // ==========================================================================
  {
    exec_place place = exec_place::current_device();

    // getStream() returns a decorated_stream with additional metadata
    decorated_stream dstream = place.getStream(resources, true);
    EXPECT(dstream.stream != nullptr);
    EXPECT(dstream.dev_id == current_device);
    EXPECT(get_device_from_stream(dstream.stream) == current_device);
  }

  // ==========================================================================
  // Test stream_pool_size() and pick_all_streams()
  // ==========================================================================
  {
    exec_place place = exec_place::current_device();

    // Query the pool size
    size_t pool_size = place.stream_pool_size(resources);
    EXPECT(pool_size > 0);
    EXPECT(pool_size == async_resources_handle::pool_size);

    // Get all streams from the pool as a vector
    auto all_streams = place.pick_all_streams(resources);
    EXPECT(all_streams.size() == pool_size);

    // Verify all streams are valid and on the correct device
    for (cudaStream_t s : all_streams)
    {
      EXPECT(s != nullptr);
      EXPECT(get_device_from_stream(s) == current_device);
    }
  }

  // ==========================================================================
  // Test with multiple devices
  // ==========================================================================
  if (device_count > 1)
  {
    for (int test_device = 0; test_device < ::std::min(device_count, 2); ++test_device)
    {
      exec_place dev_place = exec_place::device(test_device);

      // pick_stream on a specific device
      cudaStream_t stream = dev_place.pick_stream(resources);
      EXPECT(stream != nullptr);
      EXPECT(get_device_from_stream(stream) == test_device);

      // getStream returns more metadata
      decorated_stream dstream = dev_place.getStream(resources, true);
      EXPECT(dstream.stream != nullptr);
      EXPECT(dstream.dev_id == test_device);
    }
  }

  // ==========================================================================
  // Test activate()/deactivate() - generic alternative to cudaSetDevice
  // These methods can be used without a CUDASTF context
  // ==========================================================================
  {
    // Save initial device
    int initial_device;
    cuda_safe_call(cudaGetDevice(&initial_device));

    // Use activate() to switch to current device (no-op but verifies it works)
    exec_place current_place = exec_place::current_device();
    exec_place prev          = current_place.activate();

    int after_activate;
    cuda_safe_call(cudaGetDevice(&after_activate));
    EXPECT(after_activate == initial_device);

    // Restore (also a no-op in this case)
    current_place.deactivate(prev);
  }

  // Test activate()/deactivate() with multiple devices
  if (device_count > 1)
  {
    // Save initial device
    int initial_device;
    cuda_safe_call(cudaGetDevice(&initial_device));

    // Switch to device 1
    exec_place place1 = exec_place::device(1);
    exec_place prev   = place1.activate();

    // Verify we're now on device 1
    int new_device;
    cuda_safe_call(cudaGetDevice(&new_device));
    EXPECT(new_device == 1);

    // Restore previous device using deactivate
    place1.deactivate(prev);

    // Verify we're back on the initial device
    int restored_device;
    cuda_safe_call(cudaGetDevice(&restored_device));
    EXPECT(restored_device == initial_device);

    // Alternative: restore by calling activate() on the previous place
    exec_place place0 = exec_place::device(0);
    prev              = place0.activate();

    cuda_safe_call(cudaGetDevice(&new_device));
    EXPECT(new_device == 0);

    // Restore by activating the previous place directly
    prev.activate();

    cuda_safe_call(cudaGetDevice(&restored_device));
    EXPECT(restored_device == initial_device);
  }

  // Test that host exec_place activate/deactivate works (no-op in practice)
  {
    exec_place host_place = exec_place::host;
    exec_place prev       = host_place.activate();
    host_place.deactivate(prev);
  }

  // ==========================================================================
  // Test context stream picking (for comparison)
  // ==========================================================================
  {
    context ctx;
    // Contexts also have pick_stream() which uses the default execution place
    cudaStream_t stream = ctx.pick_stream();
    EXPECT(stream != nullptr);
    EXPECT(get_device_from_stream(stream) == current_device);
    ctx.finalize();
  }

  // ==========================================================================
  // Test using exec_place::pick_stream with a context's async_resources
  // When working alongside a context, use ctx.async_resources() to share
  // the same stream pools between your code and the context's operations.
  // ==========================================================================
  {
    stream_ctx ctx;

    // Get a stream from a specific execution place using the context's resources
    exec_place place     = exec_place::current_device();
    cudaStream_t stream1 = place.pick_stream(ctx.async_resources());
    EXPECT(stream1 != nullptr);
    EXPECT(get_device_from_stream(stream1) == current_device);

    // This stream comes from the same pool used by ctx internally
    cudaStream_t stream2 = ctx.pick_stream();
    EXPECT(stream2 != nullptr);

    // Both methods use the same underlying stream pool
    // (streams may or may not be the same depending on round-robin selection)

    ctx.finalize();
  }

  // ==========================================================================
  // Test with graph context
  // ==========================================================================
  {
    graph_ctx gctx;
    cudaStream_t stream = gctx.pick_stream();
    EXPECT(stream != nullptr);
    EXPECT(get_device_from_stream(stream) == current_device);
    gctx.finalize();
  }

  // ==========================================================================
  // Test context with execution affinity
  // ==========================================================================
  if (device_count > 1)
  {
    context ctx;

    exec_place dev1_place = exec_place::device(1);
    ctx.set_affinity({::std::make_shared<exec_place>(dev1_place)});

    // Stream should now come from device 1's pool
    cudaStream_t affinity_stream = ctx.pick_stream();
    EXPECT(affinity_stream != nullptr);
    EXPECT(get_device_from_stream(affinity_stream) == 1);

    ctx.finalize();
  }
}
