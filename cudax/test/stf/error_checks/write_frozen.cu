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
 * @brief Ensure a write access to frozen data from a task is reported
 */

#include <cuda/experimental/__stf/stream/stream_ctx.cuh>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

using namespace cuda::experimental::stf;

int main()
{
  stream_ctx ctx;
  const int N = 16;
  int X[N];

  for (int i = 0; i < N; i++)
  {
    X[i] = i;
  }

  auto lX = ctx.logical_data(X);

  auto fX = ctx.freeze(lX, access_mode::rw, data_place::current_device());

  bool caught = false;
  try
  {
    // Illegal: a task cannot write to frozen data.
    ctx.task(lX.rw())->*[](cudaStream_t, auto) {};
  }
  catch (const ::std::logic_error& e)
  {
    caught = true;
    fprintf(stderr, "Caught expected error: %s\n", e.what());
  }

  fX.unfreeze(ctx.fence());

  // Once unfrozen, the same access is legal again.
  ctx.task(lX.rw())->*[](cudaStream_t, auto) {};
  ctx.finalize();

  return caught ? EXIT_SUCCESS : EXIT_FAILURE;
}
