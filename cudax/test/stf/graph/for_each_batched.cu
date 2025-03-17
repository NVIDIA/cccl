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
 * @brief Generate a library call from nested CUDA graphs generated using for_each_batched and algorithms
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

// Some fake library doing MATH
// template <typename context_t>
void libMATH(graph_ctx ctx, logical_data<slice<double>> x, logical_data<slice<double>> y)
{
  // We only want to have kernels with 4 CTAs to stress the system
  auto spec = par<4>(par<128>());
  ctx.launch(spec, exec_place::current_device(), x.read(), y.write()).set_symbol("MATH1")->*
    [] __device__(auto t, auto x, auto y) {
      for (auto i : t.apply_partition(shape(x)))
      {
        y(i) = cos(cos(x(i)));
      }
    };

  ctx.launch(spec, exec_place::current_device(), x.write(), y.read()).set_symbol("MATH2")->*
    [] __device__(auto t, auto x, auto y) {
      for (auto i : t.apply_partition(shape(x)))
      {
        x(i) = sin(sin(y(i)));
      };
    };
}

template <typename context_t>
void libMATH_AS_GRAPH(context_t& ctx, logical_data<slice<double>> x, logical_data<slice<double>> y)
{
  static algorithm alg;
  alg.run_as_task(libMATH, ctx, x.rw(), y.write());
}

// Some fake lib doing a SWAP
template <typename context_t>
void libSWAP(context_t& ctx, logical_data<slice<double>> x, logical_data<slice<double>> y)
{
  // We only want to have kernels with 4 CTAs to stress the system
  auto spec = par<4>(par<128>());
  ctx.launch(spec, exec_place::current_device(), x.rw(), y.rw()).set_symbol("SWAP")->*
    [] __device__(auto t, auto x, auto y) {
      for (auto i : t.apply_partition(shape(x)))
      {
        auto tmp = x(i);
        x(i)     = y(i);
        y(i)     = tmp;
      }
    };
}

template <typename context_t>
logical_data<slice<double>> libCOPY(context_t& ctx, logical_data<slice<double>> x)
{
  logical_data<slice<double>> res = ctx.logical_data(x.shape());

  // We only want to have kernels with 4 CTAs to stress the system
  auto spec = par<4>(par<128>());
  ctx.launch(spec, exec_place::current_device(), x.read(), res.write()).set_symbol("SWAP")->*
    [] __device__(auto t, auto x, auto res) {
      for (auto i : t.apply_partition(shape(x)))
      {
        res(i) = x(i);
      }
    };

  return res;
}

int main()
{
  nvtx_range r("run");

  stream_ctx ctx;

  const size_t N = 256 * 1024;
  const size_t K = 8;

  size_t BATCH_SIZE = 4;

  logical_data<slice<double>> lX[K];
  logical_data<slice<double>> lY[K];

  for (size_t i = 0; i < K; i++)
  {
    lX[i] = ctx.logical_data<double>(N);
    lY[i] = ctx.logical_data<double>(N);

    ctx.parallel_for(lX[i].shape(), lX[i].write(), lY[i].write()).set_symbol("INIT")->*
      [] __device__(size_t i, auto x, auto y) {
        x(i) = 2.0 * i + 12.0;
        y(i) = -3.0 * i + 17.0;
      };
  }

  for_each_batched<slice<double>, slice<double>>(
    context(ctx),
    K,
    BATCH_SIZE,
    [&](size_t i) {
      return std::make_tuple(lX[i].rw(), lY[i].write());
    })
      ->*[](context ctx, size_t, auto lxi, auto lyi) {
            auto tmp = libCOPY(ctx, lxi);
            libSWAP(ctx, tmp, lyi);
            libMATH_AS_GRAPH(ctx, lxi, lyi);
            libSWAP(ctx, lxi, lyi);
          };

  ctx.task_fence();

  ctx.finalize();
}
