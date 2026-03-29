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
 * @brief Test that the node hierarchy pool grows beyond its initial capacity
 *        and that freed slots are properly reused across multiple batches.
 */

#include <cuda/experimental/stf.cuh>

#include <thread>

using namespace cuda::experimental::stf;

// The initial node pool size is 16. Using more threads than that forces growth.
static constexpr int NTHREADS = 20;
static constexpr int NBATCHES = 3;
static constexpr size_t N     = 1024;

void worker(stackable_ctx sctx, int main_head, stackable_logical_data<slice<int>> ld)
{
  sctx.set_head_offset(main_head);
  sctx.push();

  sctx.parallel_for(ld.shape(), ld.write())->*[=] __device__(size_t k, auto d) {
    d(k) = 42;
  };

  sctx.pop();
}

int main()
{
  stackable_ctx sctx;

  int main_head = sctx.get_head_offset();

  ::std::vector<stackable_logical_data<slice<int>>> lds;
  for (int i = 0; i < NTHREADS; i++)
  {
    lds.push_back(sctx.logical_data(shape_of<slice<int>>(N)));
  }

  // Run multiple batches: the first batch forces growth beyond 16,
  // subsequent batches verify that freed slots are reused.
  for (int batch = 0; batch < NBATCHES; batch++)
  {
    ::std::vector<::std::thread> threads;
    for (int i = 0; i < NTHREADS; i++)
    {
      threads.emplace_back(worker, sctx, main_head, lds[i]);
    }
    for (auto& t : threads)
    {
      t.join();
    }
  }

  sctx.finalize();

  return 0;
}
