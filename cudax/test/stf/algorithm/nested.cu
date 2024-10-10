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

  auto a = ctx.logical_data<int>(size_t(1000000));
  auto b = ctx.logical_data<int>(size_t(1000000));
  auto c = ctx.logical_data<int>(size_t(1000000));
  auto d = ctx.logical_data<int>(size_t(1000000));

  init(ctx, a, 12);
  init(ctx, b, 35);
  init(ctx, c, 42);
  init(ctx, d, 17);

  auto fn1 = [](context ctx, logical_data<slice<int>> a) {
    ctx.parallel_for(a.shape(), a.rw())->*[] __device__(size_t i, auto sa) {
      sa(i) += 1;
    };
  };

  algorithm alg1;

  auto fn2 = [&alg1, &fn1](context ctx, logical_data<slice<int>> a, logical_data<slice<int>> b) {
    alg1.run_as_task(fn1, ctx, a.rw());
    alg1.run_as_task(fn1, ctx, b.rw());
    ctx.parallel_for(a.shape(), a.rw(), b.read())->*[] __device__(size_t i, auto sa, auto sb) {
      sa(i) += sb(i);
    };
    ctx.parallel_for(a.shape(), a.read(), b.write())->*[] __device__(size_t i, auto sa, auto sb) {
      sb(i) = sa(i);
    };
  };

  algorithm alg2;

  for (size_t i = 0; i < 100; i++)
  {
    alg2.run_as_task(fn2, ctx, a.rw(), b.rw());
    alg2.run_as_task(fn2, ctx, a.rw(), c.rw());
    alg2.run_as_task(fn2, ctx, c.rw(), d.rw());
    alg2.run_as_task(fn2, ctx, d.rw(), a.rw());
  }

  ctx.finalize();
}
