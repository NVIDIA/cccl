//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <cuda/experimental/green_context.cuh>
#include <cuda/experimental/stream.cuh>

#include <testing.cuh>
#include <utility.cuh>

#if _CCCL_CTK_AT_LEAST(12, 5)
C2H_TEST("Green context", "[green_context]")
{
  if (test::cuda_driver_version() < 12050)
  {
    SUCCEED("Driver is too old for green context tests");
  }
  else
  {
    INFO("Can create a green context");
    {
      {
        [[maybe_unused]] cudax::green_context ctx(cuda::devices[0]);
      }
      {
        cudax::green_context ctx(cuda::devices[0]);
        auto handle     = ctx.release();
        auto new_object = cudax::green_context::from_native_handle(handle);
      }
    }

    INFO("Can create streams under green context");
    {
      cudax::green_context green_ctx_dev0(cuda::devices[0]);
      cudax::stream stream_under_green_ctx(green_ctx_dev0);
      CUDAX_REQUIRE(stream_under_green_ctx.device() == 0);
      if (cuda::devices.size() > 1)
      {
        cudax::green_context green_ctx_dev1(cuda::devices[1]);
        cudax::stream stream_dev1(green_ctx_dev1);
        CUDAX_REQUIRE(stream_dev1.device() == 1);
      }

      INFO("Can create a side stream");
      {
        auto ldev1 = stream_under_green_ctx.logical_device();
        CUDAX_REQUIRE(ldev1.kind() == cudax::logical_device::kinds::green_context);
        cudax::stream side_stream(ldev1);
        CUDAX_REQUIRE(side_stream.device() == 0);
        auto ldev2 = side_stream.logical_device();
        CUDAX_REQUIRE(ldev2.kind() == cudax::logical_device::kinds::green_context);
        CUDAX_REQUIRE(ldev1 == ldev2);
      }
    }
  }

#  if _CCCL_CTK_AT_LEAST(13, 0)
  if (test::cuda_driver_version() >= 13000)
  {
    INFO("Can get green context ID");
    {
      STATIC_REQUIRE(cuda::std::is_same_v<unsigned long long, cuda::std::underlying_type_t<cudax::green_context_id>>);
      STATIC_REQUIRE(
        cuda::std::is_same_v<cudax::green_context_id, decltype(cuda::std::declval<cudax::green_context>().id())>);

      cudax::green_context ctx1{cuda::devices[0]};
      cudax::green_context ctx2{cuda::devices[0]};

      // Test that id() returns a valid ID
      auto id1 = ctx1.id();
      auto id2 = ctx2.id();

      // Test that different contexts have different IDs
      CUDAX_REQUIRE(id1 != id2);

      // Test that the same context returns the same ID when called multiple times
      CUDAX_REQUIRE(ctx1.id() == id1);
      CUDAX_REQUIRE(ctx2.id() == id2);
    }
  }
#  endif // _CCCL_CTK_AT_LEAST(13, 0)
}
#else // ^^^ _CCCL_CTK_AT_LEAST(12, 5) ^^^ / vvv _CCCL_CTK_BELOW(12, 5) vvv
// For some reason CI fails with empty test, add a dummy test case
C2H_TEST("Dummy test case", "")
{
  CUDAX_REQUIRE(1 == 1);
}
#endif // ^^^ _CCCL_CTK_BELOW(12, 5) ^^^
