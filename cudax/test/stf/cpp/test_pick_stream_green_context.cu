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
//! \brief Test the pick_stream functionality with green contexts

#include <cuda/experimental/__stf/places/exec/green_context.cuh>
#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

// Green contexts are only supported since CUDA 12.4
#if _CCCL_CTK_AT_LEAST(12, 4)

//! \brief Verify that a stream belongs to the expected green context
void verify_stream_green_context(cudaStream_t stream, CUgreenCtx expected_g_ctx)
{
  // Get the green context associated to that CUDA stream
  CUgreenCtx stream_cugc;
  cuda_safe_call(cuStreamGetGreenCtx(CUstream(stream), &stream_cugc));
  EXPECT(stream_cugc != nullptr);

  CUcontext stream_green_primary;
  CUcontext expected_green_primary;

  unsigned long long stream_ctxId;
  unsigned long long expected_ctxId;

  // Convert green contexts to primary contexts and get their ID
  cuda_safe_call(cuCtxFromGreenCtx(&stream_green_primary, stream_cugc));
  cuda_safe_call(cuCtxGetId(stream_green_primary, &stream_ctxId));

  cuda_safe_call(cuCtxFromGreenCtx(&expected_green_primary, expected_g_ctx));
  cuda_safe_call(cuCtxGetId(expected_green_primary, &expected_ctxId));

  // Make sure the stream belongs to the same green context as expected
  EXPECT(stream_ctxId == expected_ctxId);
}

#endif // _CCCL_CTK_AT_LEAST(12, 4)

int main()
{
#if _CCCL_CTK_BELOW(12, 4)
  // Green contexts are not supported, skip the test
  return 0;
#else // ^^^ _CCCL_CTK_BELOW(12, 4) ^^^ / vvv _CCCL_CTK_AT_LEAST(12, 4) vvv

  // Get current device
  int current_device;
  cuda_safe_call(cudaGetDevice(&current_device));

  // Create green context helper with 8 SMs per context
  const int num_sms = 8;
  green_context_helper gc(num_sms, current_device);

  // Create async_resources_handle for stream pool management.
  // This can be used independently of any CUDASTF context.
  async_resources_handle resources;

  // ==========================================================================
  // Compare regular device vs green context execution places
  // ==========================================================================
  exec_place regular_device_place = exec_place::current_device();

  // pick_stream() returns cudaStream_t directly
  cudaStream_t device_stream = regular_device_place.pick_stream(resources);
  EXPECT(device_stream != nullptr);
  EXPECT(get_device_from_stream(device_stream) == current_device);

  // ==========================================================================
  // Test green context execution places - each has isolated stream pools
  // ==========================================================================
  auto cnt = gc.get_count();
  if (cnt > 0)
  {
    // Test first green context view - demonstrates place-specific stream pools
    auto view0           = gc.get_view(0);
    exec_place gc_place0 = exec_place::green_ctx(view0);

    // Green context execution place uses its dedicated stream pool (not shared device pool)
    cudaStream_t gc_stream = gc_place0.pick_stream(resources);
    EXPECT(gc_stream != nullptr);
    EXPECT(get_device_from_stream(gc_stream) == current_device);

    // Verify the stream belongs to the correct green context
    verify_stream_green_context(gc_stream, view0.g_ctx);

    // Test with multiple views - demonstrates isolation between green contexts
    if (cnt > 1)
    {
      auto view1           = gc.get_view(1);
      exec_place gc_place1 = exec_place::green_ctx(view1);

      cudaStream_t gc_stream1 = gc_place1.pick_stream(resources);
      EXPECT(gc_stream1 != nullptr);
      EXPECT(get_device_from_stream(gc_stream1) == current_device);

      // Each green context has its own isolated stream pool
      verify_stream_green_context(gc_stream1, view1.g_ctx);

      // Streams from different green context places are isolated
      EXPECT(gc_stream != gc_stream1);
    }

    // getStream() provides additional metadata if needed
    decorated_stream dstream = gc_place0.getStream(resources, true);
    EXPECT(dstream.stream != nullptr);
    EXPECT(dstream.dev_id == current_device);
  }

  // ==========================================================================
  // Test activate()/deactivate() with green contexts
  // These methods can be used without a CUDASTF context
  // ==========================================================================
  if (cnt > 0)
  {
    auto view           = gc.get_view(0);
    exec_place gc_place = exec_place::green_ctx(view);

    // Save the current CUDA context
    CUcontext initial_ctx;
    cuda_safe_call(cuCtxGetCurrent(&initial_ctx));

    // Activate the green context - this sets cuCtxSetCurrent to the green context
    exec_place prev = gc_place.activate();

    // Verify the current context is now the green context
    CUcontext current_ctx;
    cuda_safe_call(cuCtxGetCurrent(&current_ctx));

    // The current context should be the green context's driver context
    CUcontext green_driver_ctx;
    cuda_safe_call(cuCtxFromGreenCtx(&green_driver_ctx, view.g_ctx));

    unsigned long long current_ctx_id, green_ctx_id;
    cuda_safe_call(cuCtxGetId(current_ctx, &current_ctx_id));
    cuda_safe_call(cuCtxGetId(green_driver_ctx, &green_ctx_id));
    EXPECT(current_ctx_id == green_ctx_id);

    // Restore the previous context
    gc_place.deactivate(prev);

    // Verify we're back to the initial context
    CUcontext restored_ctx;
    cuda_safe_call(cuCtxGetCurrent(&restored_ctx));

    unsigned long long initial_ctx_id, restored_ctx_id;
    cuda_safe_call(cuCtxGetId(initial_ctx, &initial_ctx_id));
    cuda_safe_call(cuCtxGetId(restored_ctx, &restored_ctx_id));
    EXPECT(initial_ctx_id == restored_ctx_id);
  }

  // Test switching between multiple green contexts
  if (cnt > 1)
  {
    auto view0           = gc.get_view(0);
    auto view1           = gc.get_view(1);
    exec_place gc_place0 = exec_place::green_ctx(view0);
    exec_place gc_place1 = exec_place::green_ctx(view1);

    // Activate first green context
    exec_place prev0 = gc_place0.activate();

    // Verify we're in green context 0
    CUcontext current_ctx;
    cuda_safe_call(cuCtxGetCurrent(&current_ctx));
    CUcontext green0_ctx;
    cuda_safe_call(cuCtxFromGreenCtx(&green0_ctx, view0.g_ctx));
    unsigned long long current_id, green0_id;
    cuda_safe_call(cuCtxGetId(current_ctx, &current_id));
    cuda_safe_call(cuCtxGetId(green0_ctx, &green0_id));
    EXPECT(current_id == green0_id);

    // Switch to second green context
    exec_place prev1 = gc_place1.activate();

    // Verify we're now in green context 1
    cuda_safe_call(cuCtxGetCurrent(&current_ctx));
    CUcontext green1_ctx;
    cuda_safe_call(cuCtxFromGreenCtx(&green1_ctx, view1.g_ctx));
    unsigned long long green1_id;
    cuda_safe_call(cuCtxGetId(current_ctx, &current_id));
    cuda_safe_call(cuCtxGetId(green1_ctx, &green1_id));
    EXPECT(current_id == green1_id);

    // Restore to green context 0
    gc_place1.deactivate(prev1);

    cuda_safe_call(cuCtxGetCurrent(&current_ctx));
    cuda_safe_call(cuCtxGetId(current_ctx, &current_id));
    EXPECT(current_id == green0_id);

    // Restore to original context
    gc_place0.deactivate(prev0);
  }

  // ==========================================================================
  // Test context with green context affinity
  // ==========================================================================
  {
    stream_ctx ctx;

    if (cnt > 0)
    {
      // Set affinity to green context execution place
      auto view           = gc.get_view(0);
      exec_place gc_place = exec_place::green_ctx(view);

      ctx.set_affinity({::std::make_shared<exec_place>(gc_place)});

      // Context pick_stream() respects the green context affinity
      cudaStream_t stream = ctx.pick_stream();
      EXPECT(stream != nullptr);
      EXPECT(get_device_from_stream(stream) == current_device);

      // Verify stream belongs to the green context we set as affinity
      verify_stream_green_context(stream, view.g_ctx);
    }

    ctx.finalize();
  }

  // ==========================================================================
  // Test graph context with green context affinity
  // ==========================================================================
  {
    graph_ctx gctx;

    if (cnt > 0)
    {
      // Set green context affinity for graph context
      auto view           = gc.get_view(0);
      exec_place gc_place = exec_place::green_ctx(view);

      gctx.set_affinity({::std::make_shared<exec_place>(gc_place)});

      // Graph context also respects the execution place abstraction
      cudaStream_t graph_stream = gctx.pick_stream();
      EXPECT(graph_stream != nullptr);
      EXPECT(get_device_from_stream(graph_stream) == current_device);

      // Verify graph submission stream also respects green context affinity
      verify_stream_green_context(graph_stream, view.g_ctx);
    }

    gctx.finalize();
  }

  return 0;
#endif // ^^^ _CCCL_CTK_AT_LEAST(12, 4) ^^^
}
