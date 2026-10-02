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
 * @brief A task body that throws while its stream is being captured must leave the stream
 *        usable: the capture is ended on the way out, and the context keeps working.
 */

#include <cuda/experimental/__stf/graph/graph_ctx.cuh>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

using namespace cuda::experimental::stf;

int main()
{
  graph_ctx ctx;

  int X[16];
  for (int i = 0; i < 16; i++)
  {
    X[i] = i;
  }
  auto lX = ctx.logical_data(X);

  int caught = 0;

  // Typed task: the stream-taking body runs under stream capture.
  try
  {
    ctx.task(lX.rw())->*[](cudaStream_t, auto) {
      throw ::std::runtime_error("typed body");
    };
  }
  catch (const ::std::runtime_error& e)
  {
    caught++;
    fprintf(stderr, "Caught expected error: %s\n", e.what());
  }

  // Untyped task, same capture path.
  try
  {
    auto t = ctx.task();
    t.add_deps(lX.rw());
    t->*[](cudaStream_t) {
      throw ::std::runtime_error("untyped body");
    };
  }
  catch (const ::std::runtime_error& e)
  {
    caught++;
    fprintf(stderr, "Caught expected error: %s\n", e.what());
  }

  // Both captures were ended: the same data and streams still work.
  ctx.parallel_for(lX.shape(), lX.rw())->*[] __device__(size_t i, auto x) {
    x(i) *= 2;
  };
  ctx.finalize();

  for (int i = 0; i < 16; i++)
  {
    if (X[i] != 2 * i)
    {
      fprintf(stderr, "X[%d] = %d, expected %d\n", i, X[i], 2 * i);
      return EXIT_FAILURE;
    }
  }

  return caught == 2 ? EXIT_SUCCESS : EXIT_FAILURE;
}
