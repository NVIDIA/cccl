//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/__stream/stream.cuh>
#include <cuda/experimental/__utility/ensure_current_device.cuh>
#include <cuda/experimental/event.cuh>
#include <cuda/experimental/launch.cuh>

#include <utility.cuh>

namespace driver = cuda::experimental::detail::driver;

void recursive_check_device_setter(int id)
{
  int cudart_id;
  cudax::__ensure_current_device setter(cudax::device_ref{id});
  CUDAX_REQUIRE(test::count_driver_stack() == cudax::devices.size() - id);
  auto ctx = driver::ctxGetCurrent();
  CUDART(cudaGetDevice(&cudart_id));
  CUDAX_REQUIRE(cudart_id == id);

  if (id != 0)
  {
    recursive_check_device_setter(id - 1);

    CUDAX_REQUIRE(test::count_driver_stack() == cudax::devices.size() - id);
    CUDAX_REQUIRE(ctx == driver::ctxGetCurrent());
    CUDART(cudaGetDevice(&cudart_id));
    CUDAX_REQUIRE(cudart_id == id);
  }
}

TEST_CASE("ensure current device", "[device]")
{
  test::empty_driver_stack();
  // If possible use something different than CUDART default 0
  int target_device = static_cast<int>(cudax::devices.size() - 1);
  int dev_id        = 0;

  SECTION("device setter")
  {
    recursive_check_device_setter(target_device);

    CUDAX_REQUIRE(test::count_driver_stack() == 0);
  }

  SECTION("stream interactions with driver stack")
  {
    {
      cudax::stream stream(target_device);
      CUDAX_REQUIRE(test::count_driver_stack() == 0);
      {
        cudax::__ensure_current_device setter(cudax::device_ref{target_device});
        CUDAX_REQUIRE(driver::ctxGetCurrent() == driver::streamGetCtx(stream.get()));
      }
      {
        auto ev = stream.record_event();
        CUDAX_REQUIRE(test::count_driver_stack() == 0);
      }
      CUDAX_REQUIRE(test::count_driver_stack() == 0);
      {
        auto ev = stream.record_timed_event();
        CUDAX_REQUIRE(test::count_driver_stack() == 0);
      }
      {
        auto lambda = [&](int dev_id) {
          cudax::stream another_stream(dev_id);
          CUDAX_REQUIRE(test::count_driver_stack() == 0);
          stream.wait(another_stream);
          CUDAX_REQUIRE(test::count_driver_stack() == 0);
          another_stream.wait(stream);
          CUDAX_REQUIRE(test::count_driver_stack() == 0);
        };
        lambda(target_device);
        if (cudax::devices.size() > 1)
        {
          lambda(0);
        }
      }

      cudax::__ensure_current_device setter(stream);
      CUDAX_REQUIRE(test::count_driver_stack() == 1);
      CUDART(cudaGetDevice(&dev_id));
      CUDAX_REQUIRE(dev_id == target_device);
      CUDAX_REQUIRE(driver::ctxGetCurrent() == driver::streamGetCtx(stream.get()));
    }

    CUDAX_CHECK(test::count_driver_stack() == 0);

    {
      // Check NULL stream ref is handled ok
      cudax::__ensure_current_device setter1(cudax::device_ref{target_device});
      cudaStream_t null_stream = nullptr;
      auto ref                 = cuda::stream_ref(null_stream);
      auto ctx                 = driver::ctxGetCurrent();
      CUDAX_REQUIRE(test::count_driver_stack() == 1);

      cudax::__ensure_current_device setter2(ref);
      CUDAX_REQUIRE(test::count_driver_stack() == 2);
      CUDAX_REQUIRE(ctx == driver::ctxGetCurrent());
      CUDART(cudaGetDevice(&dev_id));
      CUDAX_REQUIRE(dev_id == target_device);
    }
  }

  SECTION("event interactions with driver stack")
  {
    {
      cudax::stream stream(target_device);
      CUDAX_REQUIRE(test::count_driver_stack() == 0);

      cudax::event event(stream);
      CUDAX_REQUIRE(test::count_driver_stack() == 0);

      event.record(stream);
      CUDAX_REQUIRE(test::count_driver_stack() == 0);
    }
    CUDAX_REQUIRE(test::count_driver_stack() == 0);
  }

  SECTION("launch interactions with driver stack")
  {
    cudax::stream stream(target_device);
    CUDAX_REQUIRE(test::count_driver_stack() == 0);
    cudax::launch(stream, cudax::make_hierarchy(cudax::block_dims<1>(), cudax::grid_dims<1>()), test::empty_kernel{});
    CUDAX_REQUIRE(test::count_driver_stack() == 0);
  }
}
