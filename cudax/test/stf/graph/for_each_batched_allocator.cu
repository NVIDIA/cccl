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
 * @brief Ensure the allocator used in algorithms with for_each_batched are deferred to the stream_ctx allocator
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

// x(i) += 3*i-1
template <typename context_t>
void func(context_t& ctx, logical_data<slice<double>> lx)
{
  auto tmp = ctx.logical_data(lx.shape());
  // tmp = 3*i-1
  ctx.parallel_for(tmp.shape(), tmp.write())->*[] __device__(size_t i, auto t) {
    t(i) = 3.0 * i - 1.0;
  };

  // x += tmp
  ctx.parallel_for(lx.shape(), lx.rw(), tmp.read())->*[] __device__(size_t i, auto x, auto t) {
    x(i) += t(i);
  };
}

int main()
{
  nvtx_range r("run");

  stream_ctx ctx;

  const size_t N = 256 * 1024;
  const size_t K = 8;

  size_t BATCH_SIZE = 4;

  logical_data<slice<double>> lX[K];

  for (size_t i = 0; i < K; i++)
  {
    lX[i] = ctx.logical_data<double>(N);

    ctx.parallel_for(lX[i].shape(), lX[i].write()).set_symbol("INIT")->*[] __device__(size_t i, auto x) {
      x(i) = 2.0 * i + 12.0;
    };
  }

  for_each_batched<slice<double>>(
    context(ctx),
    K,
    BATCH_SIZE,
    [&](size_t i) {
      return lX[i].rw();
    })
      ->*[&](context ctx, size_t, auto lxi) {
            // We use a function not to inline extended lambdas within a lambda
            func(ctx, lxi);
          };

  for (size_t i = 0; i < K; i++)
  {
    ctx.host_launch(lX[i].read()).set_symbol("check")->*[](auto x) {
      for (size_t ind = 0; ind < N; ind++)
      {
        double expected = 2.0 * ind + 12.0 + 3.0 * ind - 1.0;
        _CCCL_ASSERT(fabs(x(ind) - expected) < 0.01, "invalid result");
      }
    };
  }

  ctx.finalize();
}
