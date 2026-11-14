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
 * @brief Ensure an error is reported when a task uses a logical data from a
 *        different context
 */

#include <cuda/experimental/__stf/graph/graph_ctx.cuh>
#include <cuda/experimental/__stf/stream/stream_ctx.cuh>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

using namespace cuda::experimental::stf;

template <typename Ctx, size_t n>
bool run(double (&X)[n])
{
  Ctx ctx1;
  auto lX = ctx1.logical_data(X);

  Ctx ctx2;

  bool caught = false;
  try
  {
    // lX belongs to ctx1: a task of ctx2 cannot use it.
    ctx2.task(lX.rw())->*[&](cudaStream_t /*unused*/, auto /*unused*/) {};
  }
  catch (const ::std::invalid_argument& e)
  {
    caught = true;
    fprintf(stderr, "Caught expected error: %s\n", e.what());
  }

  // Both contexts are still usable.
  ctx1.task(lX.rw())->*[&](cudaStream_t /*unused*/, auto /*unused*/) {};
  ctx2.finalize();
  ctx1.finalize();

  return caught;
}

int main()
{
  const int n = 12;
  double X[n];

  for (int ind = 0; ind < n; ind++)
  {
    X[ind] = 1.0 * ind;
  }

  const bool ok = run<stream_ctx>(X) && run<graph_ctx>(X);
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
