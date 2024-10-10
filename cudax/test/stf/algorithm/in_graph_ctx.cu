//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

template <typename context_t, typename T>
void init(context_t& ctx, logical_data<T> l, int val)
{
  ctx.parallel_for(l.shape(), l.write())->*[=] __device__(size_t i, auto s) {
    s(i) = val;
  };
}

int main()
{
  //    context ctx = graph_ctx();
  graph_ctx ctx;

  auto a = ctx.logical_data<int>(size_t(1000000));
  auto b = ctx.logical_data<int>(size_t(1000000));
  auto c = ctx.logical_data<int>(size_t(1000000));
  auto d = ctx.logical_data<int>(size_t(1000000));

  init(ctx, a, 12);
  init(ctx, b, 35);
  init(ctx, c, 42);
  init(ctx, d, 42);

  auto fn = [](context ctx, logical_data<slice<int>> a, logical_data<slice<int>> b) {
    ctx.parallel_for(a.shape(), a.rw())->*[] __device__(size_t i, auto sa) {
      sa(i) *= 3;
    };
    ctx.parallel_for(b.shape(), b.rw())->*[] __device__(size_t i, auto sb) {
      sb(i) *= 2;
    };

    ctx.parallel_for(a.shape(), a.rw(), b.read())->*[] __device__(size_t i, auto sa, auto sb) {
      sa(i) += sb(i);
    };

    ctx.parallel_for(a.shape(), a.read(), b.write())->*[] __device__(size_t i, auto sa, auto sb) {
      sb(i) = sa(i);
    };
  };

  algorithm alg;

  for (size_t i = 0; i < 5; i++)
  {
    alg.run_as_task(fn, ctx, a.rw(), b.rw());
    alg.run_as_task(fn, ctx, a.rw(), c.rw());
    alg.run_as_task(fn, ctx, c.rw(), d.rw());
    alg.run_as_task(fn, ctx, d.rw(), a.rw());
  }

  ctx.finalize();

  // cudaGraphDebugDotPrint(ctx.get_graph(), "pif.dot", 0);
}
