//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Ensure an error is reported if we use an uninitialized logical data in a task
 */

#include <cuda/experimental/__stf/graph/graph_ctx.cuh>
#include <cuda/experimental/__stf/stream/stream_ctx.cuh>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

using namespace cuda::experimental::stf;

template <typename Ctx>
bool run()
{
  Ctx ctx;

  int X[128];
  auto lX = ctx.logical_data(X);

  // Never initialized: using it as a dependency is a programming error.
  logical_data<slice<int>> lY;

  bool caught = false;
  try
  {
    ctx.task(lX.rw(), lY.rw())->*[](cudaStream_t, auto, auto) {};
  }
  catch (const ::std::invalid_argument& e)
  {
    caught = true;
    fprintf(stderr, "Caught expected error: %s\n", e.what());
  }

  // The rejected task left the context usable.
  ctx.task(lX.rw())->*[](cudaStream_t, auto) {};
  ctx.finalize();

  return caught;
}

int main()
{
  const bool ok = run<stream_ctx>() && run<graph_ctx>();
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
