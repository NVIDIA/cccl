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
 * @brief Ensure an error is reported if we try to ask for an unreasonable
 *        amount of resources in a thread hierarchy spec
 */

#include <cuda/experimental/stf.cuh>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

using namespace cuda::experimental::stf;

int main()
{
  context ctx;

  int X[128];
  auto lX = ctx.logical_data(X);

  bool caught = false;
  try
  {
    // We are asking an unreasonable amount of threads per block
    auto spec = con(con<128000>());
    ctx.launch(spec, lX.rw())->*[] __device__(auto th, auto X) {
      X[th.rank()] = th.rank();
    };
  }
  catch (const ::std::invalid_argument& e)
  {
    caught = true;
    fprintf(stderr, "Caught expected error: %s\n", e.what());
  }

  // A satisfiable spec still works afterwards.
  ctx.launch(con(con<128>()), lX.rw())->*[] __device__(auto th, auto X) {
    X[th.rank()] = th.rank();
  };
  ctx.finalize();

  return caught ? EXIT_SUCCESS : EXIT_FAILURE;
}
