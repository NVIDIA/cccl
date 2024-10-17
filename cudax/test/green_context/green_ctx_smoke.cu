//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/green_context.cuh>
#include <cuda/experimental/stream.cuh>

#include <catch2/catch.hpp>
#include <utility.cuh>

#if CUDART_VERSION >= 12050
TEST_CASE("Green context", "[green_context]")
{
  if (test::cuda_driver_version() < 12050)
  {
    SUCCEED("Driver is too old for green context tests");
  }
  else
  {
    INFO("Can create a green context")
    {
      {
        [[maybe_unused]] cudax::green_context ctx(cudax::devices[0]);
      }
      {
        cudax::green_context ctx(cudax::devices[0]);
        auto handle     = ctx.release();
        auto new_object = cudax::green_context::from_native_handle(handle);
      }
    }

    INFO("Can create streams under green context")
    {
      cudax::green_context green_ctx_dev0(cudax::devices[0]);
      cudax::stream stream_under_green_ctx(green_ctx_dev0);
      CUDAX_REQUIRE(stream_under_green_ctx.device() == 0);
      if (cudax::devices.size() > 1)
      {
        cudax::green_context green_ctx_dev1(cudax::devices[1]);
        cudax::stream stream_dev1(green_ctx_dev1);
        CUDAX_REQUIRE(stream_dev1.device() == 1);
      }

      INFO("Can create a side stream")
      {
        auto ldev1 = stream_under_green_ctx.logical_device();
        CUDAX_REQUIRE(ldev1.get_kind() == cudax::logical_device::kinds::green_context);
        cudax::stream side_stream(ldev1);
        CUDAX_REQUIRE(side_stream.device() == 0);
        auto ldev2 = side_stream.logical_device();
        CUDAX_REQUIRE(ldev2.get_kind() == cudax::logical_device::kinds::green_context);
        CUDAX_REQUIRE(ldev1 == ldev2);
      }
    }
  }
}
#endif // CUDART_VERSION >= 12050
