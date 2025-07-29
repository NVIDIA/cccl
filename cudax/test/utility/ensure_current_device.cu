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
#include <cuda/experimental/launch.cuh>

#include <utility.cuh>

namespace driver = cuda::__driver;

void recursive_check_device_setter(int id)
{
  int cudart_id;
  cudax::__ensure_current_device setter(cuda::device_ref{id});
  CUDAX_REQUIRE(test::count_driver_stack() == cuda::devices.size() - id);
  auto ctx = driver::__ctxGetCurrent();
  CUDART(cudaGetDevice(&cudart_id));
  CUDAX_REQUIRE(cudart_id == id);

  if (id != 0)
  {
    recursive_check_device_setter(id - 1);

    CUDAX_REQUIRE(test::count_driver_stack() == cuda::devices.size() - id);
    CUDAX_REQUIRE(ctx == driver::__ctxGetCurrent());
    CUDART(cudaGetDevice(&cudart_id));
    CUDAX_REQUIRE(cudart_id == id);
  }
}

C2H_TEST("ensure current device", "[device]")
{
  test::empty_driver_stack();
  // If possible use something different than CUDART default 0
  int target_device = static_cast<int>(cuda::devices.size() - 1);

  SECTION("device setter")
  {
    recursive_check_device_setter(target_device);

    CUDAX_REQUIRE(test::count_driver_stack() == 0);
  }
}
