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

template <typename T>
void init(context& ctx, logical_data<T> l, int val)
{
  ctx.parallel_for(l.shape(), l.write())->*[=] __device__(size_t i, auto s) {
    s(i) = val;
  };
}

int main()
{
  context ctx;

  auto a = ctx.logical_data<int>(10000000);
  auto b = ctx.logical_data<int>(10000000);
  auto c = ctx.logical_data<int>(10000000);
  auto d = ctx.logical_data<int>(10000000);

  init(ctx, a, 12);
  init(ctx, b, 35);
  init(ctx, c, 42);
  init(ctx, d, 17);

  /* a += 1; a += b; */
  auto fn = [](context ctx, logical_data<slice<int>> a, logical_data<slice<int>> b) {
    ctx.parallel_for(a.shape(), a.rw())->*[] __device__(size_t i, auto sa) {
      sa(i) += 1;
    };
    ctx.parallel_for(a.shape(), a.rw(), b.read())->*[] __device__(size_t i, auto sa, auto sb) {
      sa(i) += sb(i);
    };
  };

  algorithm alg;

  for (size_t i = 0; i < 100; i++)
  {
    alg.run_as_task(fn, ctx, a.rw(), b.read());
    alg.run_as_task(fn, ctx, a.rw(), c.read());
    alg.run_as_task(fn, ctx, c.rw(), d.read());
    alg.run_as_task(fn, ctx, d.rw(), a.read());
  }

  ctx.finalize();
}
