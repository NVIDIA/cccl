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

template <typename ctx_t, typename T, typename T2>
void lib_call(ctx_t& ctx, logical_data<T> a, logical_data<T2> b)
{
  nvtx_range r("lib_call");
  // b *= 2
  // a = a + b
  // b = a
  ctx.parallel_for(b.shape(), b.rw())->*[] __device__(size_t i, auto sb) {
    sb(i) *= 2;
  };

  ctx.parallel_for(a.shape(), a.rw(), b.read())->*[] __device__(size_t i, auto sa, auto sb) {
    sa(i) += sb(i);
  };

  ctx.parallel_for(a.shape(), a.read(), b.write())->*[] __device__(size_t i, auto sa, auto sb) {
    sb(i) = sa(i);
  };
}

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

  auto a = ctx.logical_data<int>(size_t(10000000));
  auto b = ctx.logical_data<int>(size_t(10000000));
  auto c = ctx.logical_data<int>(size_t(10000000));
  auto d = ctx.logical_data<int>(size_t(10000000));

  init(ctx, a, 12);
  init(ctx, b, 35);
  init(ctx, c, 42);
  init(ctx, d, 42);

  auto fn = [](context ctx, logical_data<slice<int>> a, logical_data<slice<int>> b) {
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

  {
    nvtx_range r("run");
    for (size_t i = 0; i < 100; i++)
    {
      ctx.task(a.rw(), b.rw())->*[&alg, &fn, &ctx](cudaStream_t stream, slice<int> sa, slice<int> sb) {
        alg.run(fn, ctx, stream, sa, sb);
      };

      ctx.task(b.rw(), c.rw())->*[&alg, &fn, &ctx](cudaStream_t stream, slice<int> sb, slice<int> sc) {
        alg.run(fn, ctx, stream, sb, sc);
      };

      ctx.task(c.rw(), d.rw())->*[&alg, &fn, &ctx](cudaStream_t stream, slice<int> sc, slice<int> sd) {
        alg.run(fn, ctx, stream, sc, sd);
      };

      ctx.task(d.rw(), a.rw())->*[&alg, &fn, &ctx](cudaStream_t stream, slice<int> sd, slice<int> sa) {
        alg.run(fn, ctx, stream, sd, sa);
      };
    }
  }

  {
    nvtx_range r("run_as_task");
    for (size_t i = 0; i < 100; i++)
    {
      alg.run_as_task(fn, ctx, a.rw(), b.rw());

      alg.run_as_task(fn, ctx, a.rw(), c.rw());

      alg.run_as_task(fn, ctx, c.rw(), d.rw());

      alg.run_as_task(fn, ctx, d.rw(), a.rw());
    }
  }

  ctx.finalize();
}
