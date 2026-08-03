//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Composite (localized) data places inside conditional graph scopes
 *
 * Regression test: localized_array allocations cached by a nested context
 * must survive the pop -- their teardown unmaps VMM backing with synchronous
 * driver calls, so destroying them with the nested context races the body
 * graph launched by the pop (cudaErrorIllegalAddress). The cache is handed
 * over to the parent context, gated on the body's completion events.
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;
using namespace cuda::experimental::places;

int main()
{
#if _CCCL_CTK_BELOW(12, 4)
  fprintf(stderr, "Waiving test: conditional nodes are only available since CUDA 12.4.\n");
  return 0;
#else
  stackable_ctx ctx;

  // A 2-place grid on the current device: enough to build a composite data
  // place without requiring several devices or green context support.
  auto grid = exec_place::repeat(exec_place::current_device(), 2);

  const size_t nx = 404, nz = 204, nv = 4;
  const size_t iters = 30;
  const auto part    = make_partition(dim4(nx, nz, nv), partition_spec{whole, blocked<0>, whole}, grid.get_dims());
  const auto dp      = make_composite_data_place(grid, part);

  auto l = ctx.logical_data(shape_of<slice<double, 3>>(nx, nz, nv)).set_symbol("field");

  ctx.parallel_for(part, grid, l.shape(), l.write(dp)).set_symbol("init")->*
    [] __device__(size_t i, size_t k, size_t v, auto s) {
      s(i, k, v) = 1.0;
    };

  // While-form conditional scope driven by a device counter
  auto lcnt = ctx.logical_data(shape_of<scalar_view<size_t>>()).set_symbol("counter");
  ctx.parallel_for(box(1), lcnt.write()).set_symbol("init_counter")->*[iters] __device__(size_t, auto c) {
    *c = iters;
  };

  {
    auto wg = ctx.while_graph_scope();
    ctx.parallel_for(part, grid, l.shape(), l.rw(dp)).set_symbol("body")->*
      [] __device__(size_t i, size_t k, size_t v, auto s) {
        s(i, k, v) += 1.0;
      };
    wg.update_cond(lcnt.rw())->*[] __device__(auto c) {
      (*c)--;
      return (*c > 0);
    };
  }

  // Fixed-count form on the same composite field (also exercises reuse of
  // the cached localized_array imported into the parent by the first pop)
  {
    auto rg = ctx.repeat_graph_scope(iters);
    ctx.parallel_for(part, grid, l.shape(), l.rw(dp)).set_symbol("body2")->*
      [] __device__(size_t i, size_t k, size_t v, auto s) {
        s(i, k, v) += 1.0;
      };
  }

  ctx.host_launch(l.read()).set_symbol("check")->*[&](auto s) {
    for (size_t v = 0; v < nv; v++)
    {
      for (size_t k = 0; k < nz; k++)
      {
        for (size_t i = 0; i < nx; i++)
        {
          EXPECT(s(i, k, v) == 1.0 + 2.0 * (double) iters);
        }
      }
    }
  };

  ctx.finalize();
  return 0;
#endif
}
