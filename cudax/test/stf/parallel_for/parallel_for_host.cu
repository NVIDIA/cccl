//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// parallel_for does not execute on the host: `parallel_for(exec_place::host(), ...)` is a
// compile-time error, and this test records the supported way to run the same host work,
// which is host_launch with an explicit loop.

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

int main()
{
  context ctx;

  int nqpoints = 3;
  auto ltoken  = ctx.token();

  ctx.host_launch(ltoken.read())->*[nqpoints](auto...) {
    for (size_t i = 0; i < 5; i++)
    {
      _CCCL_ASSERT(nqpoints == 3, "invalid value");
    }
  };

  ctx.finalize();
}
