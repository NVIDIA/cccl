//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__runtime/ensure_current_context.h>

#include <testing.cuh>

namespace driver = cuda::__driver;

void recursive_check_device_setter(int id)
{
  int cudart_id;
  cuda::__ensure_current_context setter(cuda::device_ref{id});
  CCCLRT_REQUIRE(test::count_driver_stack() == cuda::devices.size() - id);
  auto ctx = driver::__ctxGetCurrent();
  CUDART(cudaGetDevice(&cudart_id));
  CCCLRT_REQUIRE(cudart_id == id);

  if (id != 0)
  {
    recursive_check_device_setter(id - 1);

    CCCLRT_REQUIRE(test::count_driver_stack() == cuda::devices.size() - id);
    CCCLRT_REQUIRE(ctx == driver::__ctxGetCurrent());
    CUDART(cudaGetDevice(&cudart_id));
    CCCLRT_REQUIRE(cudart_id == id);
  }
}

C2H_TEST("ensure current context", "[device]")
{
  test::empty_driver_stack();
  // If possible use something different than CUDART default 0
  int target_device = static_cast<int>(cuda::devices.size() - 1);

  SECTION("context setter")
  {
    recursive_check_device_setter(target_device);

    CCCLRT_REQUIRE(test::count_driver_stack() == 0);
  }
}
