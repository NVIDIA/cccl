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

void test_shape_from_scalar_view()
{
  double x = 0;
  scalar_view<double> sv(&x);
  shape_of<scalar_view<double>> s = shape(sv);
  EXPECT(s.size() == sizeof(double));

  size_t n = 0;
  scalar_view<size_t> sv_n(&n);
  shape_of<scalar_view<size_t>> s_n = shape(sv_n);
  EXPECT(s_n.size() == sizeof(size_t));
}

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

  // Exercise logical_data(la.shape()) when la is scalar_view-backed (uses shape_of from scalar_view)
  auto ld = ctx.logical_data(la.shape()).set_symbol("d");
  ctx.parallel_for(box(1), la.read(), ld.write())->*[] __device__(size_t, auto a, auto d) {
    *d.addr = *a.addr;
  };
  ctx.host_launch(ld.read())->*[](auto x) {
    EXPECT(fabs(*x.addr - 42.0) < 0.001);
  };

  ctx.finalize();
}

int main()
{
  test_shape_from_scalar_view();
  run<stream_ctx>();
  run<graph_ctx>();
}
