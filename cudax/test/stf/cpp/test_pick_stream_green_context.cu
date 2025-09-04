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

  // Test execution place abstraction with different place types
  async_resources_handle resources;

  // Compare regular device execution place vs green context execution places
  exec_place regular_device_place = exec_place::current_device();
  decorated_stream device_stream  = regular_device_place.getStream(resources, true);
  EXPECT(device_stream.stream != nullptr);
  EXPECT(device_stream.dev_id == current_device);

  // Test green context execution places - each has isolated stream pools
  auto cnt = gc.get_count();
  if (cnt > 0)
  {
    // Test first green context view - demonstrates place-specific stream pools
    auto view0           = gc.get_view(0);
    exec_place gc_place0 = exec_place::green_ctx(view0);

    // Green context execution place uses its dedicated stream pool (not shared device pool)
    decorated_stream gc_stream = gc_place0.getStream(resources, true);
    EXPECT(gc_stream.stream != nullptr);
    EXPECT(gc_stream.dev_id == current_device);
    EXPECT(get_device_from_stream(gc_stream.stream) == current_device);

    // Verify the stream belongs to the correct green context
    verify_stream_green_context(gc_stream.stream, view0.g_ctx);

    // Test with multiple views - demonstrates isolation between green contexts
    if (cnt > 1)
    {
      auto view1           = gc.get_view(1);
      exec_place gc_place1 = exec_place::green_ctx(view1);

      decorated_stream gc_stream1 = gc_place1.getStream(resources, true);
      EXPECT(gc_stream1.stream != nullptr);
      EXPECT(gc_stream1.dev_id == current_device);
      EXPECT(get_device_from_stream(gc_stream1.stream) == current_device);

      // Each green context has its own isolated stream pool
      verify_stream_green_context(gc_stream1.stream, view1.g_ctx);

      // Streams from different green context places are isolated
      EXPECT(gc_stream.stream != gc_stream1.stream);
    }
  }

  // Test context with green context affinity - demonstrates execution place abstraction
  {
    stream_ctx ctx;

    if (cnt > 0)
    {
      // Set affinity to green context execution place
      auto view           = gc.get_view(0);
      exec_place gc_place = exec_place::green_ctx(view);

      ctx.set_affinity({::std::make_shared<exec_place>(gc_place)});

      // Context pick_stream() uses default_exec_place().getStream() internally
      // This respects the green context affinity we just set
      cudaStream_t stream = ctx.pick_stream();
      EXPECT(stream != nullptr);
      EXPECT(get_device_from_stream(stream) == current_device);

      // Verify stream belongs to the green context we set as affinity
      // This proves the execution place abstraction is working correctly
      verify_stream_green_context(stream, view.g_ctx);
    }

    ctx.finalize();
  }

  // Test graph context with green context affinity
  {
    graph_ctx gctx;

    if (cnt > 0)
    {
      // Set green context affinity for graph context
      auto view           = gc.get_view(0);
      exec_place gc_place = exec_place::green_ctx(view);

      gctx.set_affinity({::std::make_shared<exec_place>(gc_place)});

      // Graph context also uses default_exec_place().getStream() internally
      // This demonstrates that both context types use the same execution place abstraction
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
