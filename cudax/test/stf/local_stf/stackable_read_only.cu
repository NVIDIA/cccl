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
 *
 * @brief Test set_read_only() on stackable logical data
 *
 * When data is marked read-only, it should be auto-pushed as read in nested
 * contexts, allowing concurrent reads from multiple graph scopes without
 * the conservative rw push that would serialize access.
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

int main()
{
  stackable_ctx ctx;

  constexpr size_t N = 1024;
  int host_data[N];
  for (size_t i = 0; i < N; i++)
  {
    host_data[i] = static_cast<int>(i * 3 + 7);
  }

  auto lConst = ctx.logical_data(host_data).set_symbol("const_data");
  lConst.set_read_only();

  auto lOut = ctx.logical_data(shape_of<slice<int>>(N)).set_symbol("output");

  // Use read-only data in multiple sequential graph scopes. Because the data
  // is read-only, it is auto-pushed as read (not rw), and the original host
  // buffer must remain unchanged after finalize.
  {
    auto scope = ctx.graph_scope();

    ctx.parallel_for(lOut.shape(), lOut.write(), lConst.read())->*[] __device__(size_t i, auto out, auto cst) {
      out(i) = cst(i) + 1;
    };
  }

  {
    auto scope = ctx.graph_scope();

    ctx.parallel_for(lOut.shape(), lOut.rw(), lConst.read())->*[] __device__(size_t i, auto out, auto cst) {
      out(i) += cst(i);
    };
  }

  // Third scope: use read-only data alongside the accumulated output
  {
    auto scope = ctx.graph_scope();

    ctx.parallel_for(lOut.shape(), lOut.rw(), lConst.read())->*[] __device__(size_t i, auto out, auto cst) {
      out(i) += cst(i) * 2;
    };
  }

  ctx.finalize();

  // Verify the read-only host buffer was not modified
  for (size_t i = 0; i < N; i++)
  {
    EXPECT(host_data[i] == static_cast<int>(i * 3 + 7));
  }

  return 0;
}
