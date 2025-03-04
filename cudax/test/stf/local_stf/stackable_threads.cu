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
 * @brief Experiment with local context nesting
 *
 */

#include <cuda/experimental/stf.cuh>

#include <thread>

#include "cuda/experimental/__stf/utility/stackable_ctx.cuh"

using namespace cuda::experimental::stf;

void worker(stackable_ctx sctx,
            int id,
            int main_head,
            stackable_logical_data<slice<int>> lAi,
            stackable_logical_data<slice<int>> lB)
{
  fprintf(stderr, "launching thread %d\n", id);
  sctx.set_head_offset(main_head);

  for (size_t k = 0; k < 3; k++)
  {
    sctx.push();

    sctx.parallel_for(lAi.shape(), lAi.write())->*[] __device__(size_t k, auto ai) {
      ai(k) = k;
    };

    sctx.parallel_for(lAi.shape(), lAi.rw(), lB.read())->*[] __device__(size_t k, auto ai, auto b) {
      ai(k) += int(cos(10.0 * b(k)));
    };

    sctx.pop();
  }
}

int main()
{
  const size_t N = 1024000;
  stackable_ctx sctx;

  int array[N];
  for (size_t i = 0; i < N; i++)
  {
    array[i] = 1 + i * i;
  }

  auto lB = sctx.logical_data(array);

  lB.set_read_only();

  int main_head = sctx.get_head_offset();

  ::std::vector<::std::thread> threads;
  ::std::vector<stackable_logical_data<slice<int>>> lA;

  const int NTHREADS = 4;

  for (int i = 0; i < NTHREADS; ++i)
  {
    lA.push_back(sctx.logical_data(shape_of<slice<int>>(N)));

    threads.emplace_back(worker, sctx, i, main_head, lA[i], lB);
  }

  for (int i = 0; i < NTHREADS; ++i)
  {
    threads[i].join();
  }

  sctx.finalize();
}
