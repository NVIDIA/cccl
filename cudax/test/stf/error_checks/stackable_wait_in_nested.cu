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
 * @brief Ensure wait() in a nested stackable context is reported
 */

#include <cuda/experimental/stf.cuh>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

using namespace cuda::experimental::stf;

int main()
{
  stackable_ctx sctx;

  auto lval = sctx.logical_data(shape_of<scalar_view<int>>());

  sctx.parallel_for(box(1), lval.write())->*[] __device__(size_t, auto val) {
    *val = 42;
  };

  bool caught = false;
  {
    auto scope = sctx.graph_scope();

    sctx.parallel_for(box(1), lval.rw())->*[] __device__(size_t, auto val) {
      *val += 1;
    };

    try
    {
      sctx.wait(lval); // wait() in a nested context is not supported
    }
    catch (const ::std::logic_error& e)
    {
      caught = true;
      fprintf(stderr, "Caught expected error: %s\n", e.what());
    }
  }

  // Back at the root, wait() is legal and sees the nested update.
  const int v = sctx.wait(lval);
  sctx.finalize();

  return (caught && v == 43) ? EXIT_SUCCESS : EXIT_FAILURE;
}
