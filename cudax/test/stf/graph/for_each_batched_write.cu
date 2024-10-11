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
 * @brief Ensure write-only accesses produces an output with for_each_batched
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

template <typename context_t>
void COPY(context_t& ctx, logical_data<slice<double>> x, logical_data<slice<double>> y)
{
  // We only want to have kernels with 4 CTAs to stress the system
  auto spec = par<4>(par<128>());
  ctx.launch(spec, exec_place::current_device(), x.read(), y.write()).set_symbol("COPY")->*
    [] __device__(auto t, auto x, auto y) {
      for (auto i : t.apply_partition(shape(x)))
      {
        y(i) = x(i);
      }
    };
}

int main()
{
  stream_ctx ctx;

  const size_t N = 256 * 1024;
  const size_t K = 8;

  const size_t BATCH_SIZE = 4;

  logical_data<slice<double>> lX[K];
  logical_data<slice<double>> lY[K];

  for (size_t i = 0; i < K; i++)
  {
    lX[i] = ctx.logical_data<double>(N);
    lX[i].set_symbol("x" + std::to_string(i));
    ctx.parallel_for(lX[i].shape(), lX[i].write()).set_symbol("INIT")->*[] __device__(size_t i, auto x) {
      x(i) = 2.0 * i + 12.0;
    };
  }

  for (size_t i = 0; i < K; i++)
  {
    // NOT INITIALIZED
    lY[i] = ctx.logical_data<double>(N);
    lY[i].set_symbol("y" + std::to_string(i));
  }

  for_each_batched<slice<double>, slice<double>>(
    context(ctx),
    K,
    BATCH_SIZE,
    [&](size_t i) {
      return std::make_tuple(lX[i].read(), lY[i].write());
    })
      ->*[](context inner_ctx, size_t, auto lxi, auto lyi) {
            COPY(inner_ctx, lxi, lyi);
          };

  for (size_t i = 0; i < K; i++)
  {
    // TODO check actual content
    ctx.task(lY[i].read()).set_symbol("CHECK")->*[](cudaStream_t, auto) {};
  }

  ctx.finalize();
}
