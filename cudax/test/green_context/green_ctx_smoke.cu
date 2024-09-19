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
#include <testing.cuh>

#if CUDART_VERSION >= 12050
TEST_CASE("Can create a green context", "[green_context]")
{
  {
    [[maybe_unused]] cudax::green_context ctx(cudax::devices[0]);
  }
  {
    cudax::green_context ctx(cudax::devices[0]);
    auto handle     = ctx.release();
    auto new_object = cudax::green_context::from_native_handle(handle);
  }
  cudax::green_context green_ctx_dev0(cudax::devices[0]);
  cudax::stream stream_under_green_ctx(green_ctx_dev0);
  CUDAX_REQUIRE(stream_under_green_ctx.device() == 0);
  if (cudax::devices.size() > 1)
  {
    cudax::green_context green_ctx_dev1(cudax::devices[1]);
    cudax::stream stream_dev1(green_ctx_dev1);
    CUDAX_REQUIRE(stream_dev1.device() == 1);
    auto dev_ref_v2 = stream_dev1.logical_device();
    cudax::stream another_stream_dev1(dev_ref_v2);
    CUDAX_REQUIRE(another_stream_dev1.device() == 1);
  }
}
#endif