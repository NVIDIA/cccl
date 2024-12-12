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
 *
 * @brief Ensure that the scalar data interface works on both stream and graph backends
 *
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

template <typename Ctx>
void run()
{
  Ctx ctx;

  double a = 42.0;
  double b = 12.3;

  auto la = ctx.logical_data(scalar_view<double>(&a)).set_symbol("a");
  auto lb = ctx.logical_data(scalar_view<double>(&b)).set_symbol("b");
  auto lc = ctx.logical_data(shape_of<scalar_view<double>>()).set_symbol("c");

  ctx.parallel_for(box(1), la.read(), lb.read(), lc.write())->*[] __device__(size_t, auto a, auto b, auto c) {
    *c.addr = *a.addr + *b.addr;
  };

  ctx.host_launch(lc.read())->*[](auto x) {
    EXPECT(fabs(*x.addr - (42.0 + 12.3)) < 0.001);
  };

  ctx.finalize();
}

int main()
{
  run<stream_ctx>();
  run<graph_ctx>();
}
