//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Ensure fence() in a nested stackable context is reported
 */

#include <cuda/experimental/stf.cuh>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

using namespace cuda::experimental::stf;

int main()
{
  stackable_ctx sctx;

  auto lA = sctx.logical_data(shape_of<slice<int>>(64));

  sctx.parallel_for(lA.shape(), lA.write())->*[] __device__(size_t i, auto a) {
    a(i) = static_cast<int>(i);
  };

  bool caught = false;
  {
    auto scope = sctx.graph_scope();

    sctx.parallel_for(lA.shape(), lA.rw())->*[] __device__(size_t i, auto a) {
      a(i) *= 2;
    };

    try
    {
      sctx.fence(); // fence() in a nested context is not supported
    }
    catch (const ::std::logic_error& e)
    {
      caught = true;
      fprintf(stderr, "Caught expected error: %s\n", e.what());
    }
  }

  // Back at the root, fence() is legal and the context finalizes normally.
  sctx.fence();
  sctx.finalize();

  return caught ? EXIT_SUCCESS : EXIT_FAILURE;
}
