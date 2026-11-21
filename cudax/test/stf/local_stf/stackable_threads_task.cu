//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief Concurrent nested contexts importing shared data through task()
 *
 * Same structure as stackable_threads.cu, but the shared datum is imported
 * through the task() path. task() returns a deferred builder, and the import
 * (push_at) happens when the builder is concretized, after task() has
 * returned. parallel_for()/launch()/cuda_kernel() hold the context-tree
 * shared lock across the import; concretization must do the same, otherwise a
 * concurrent push() on another thread may grow the context node vector under
 * the references taken by push_at(), and a concurrent pop() may release a
 * shared freeze that push_at() is reusing.
 */

#include <cuda/experimental/stf.cuh>

#include <thread>

using namespace cuda::experimental::stf;

void worker(
  stackable_ctx sctx, int main_head, stackable_logical_data<slice<int>> lAi, stackable_logical_data<slice<int>> lB)
{
  sctx.set_head_offset(main_head);
  sctx.push();

  auto lC = sctx.logical_data_no_export(lB.shape());

  // Import lB (shared, read-only) through the task() path.
  sctx.task(lC.write(), lB.read())->*[](cudaStream_t s, auto c, auto b) {
    cuda_safe_call(
      cudaMemcpyAsync(c.data_handle(), b.data_handle(), c.size() * sizeof(int), cudaMemcpyDeviceToDevice, s));
  };

  sctx.parallel_for(lAi.shape(), lAi.write())->*[] __device__(size_t k, auto ai) {
    ai(k) = k;
  };

  sctx.parallel_for(lAi.shape(), lAi.rw(), lC.read())->*[] __device__(size_t k, auto ai, auto c) {
    ai(k) += c(k);
  };

  sctx.pop();
}

int main()
{
  const size_t N = 1024000;

  stackable_ctx sctx;

  ::std::vector<int> array(N);
  for (size_t i = 0; i < N; i++)
  {
    array[i] = static_cast<int>(1 + i * i);
  }

  auto lB = sctx.logical_data(make_slice(array.data(), N));
  lB.set_read_only();

  const int main_head = sctx.get_head_offset();

  ::std::vector<stackable_logical_data<slice<int>>> lA;
  const int NTHREADS = 8;
  lA.reserve(NTHREADS);
  for (int i = 0; i < NTHREADS; ++i)
  {
    lA.push_back(sctx.logical_data(shape_of<slice<int>>(N)));
  }

  for (int k = 0; k < 30; k++)
  {
    ::std::vector<::std::thread> threads;
    threads.reserve(NTHREADS);
    for (int i = 0; i < NTHREADS; ++i)
    {
      threads.emplace_back(worker, sctx, main_head, lA[i], lB);
    }
    for (int i = 0; i < NTHREADS; ++i)
    {
      threads[i].join();
    }
  }

  // Check one result: ai(k) == k + b(k)
  sctx.host_launch(lA[0].read(), lB.read())->*[](auto a0, auto b) {
    for (size_t k = 0; k < a0.size(); k++)
    {
      EXPECT(a0(k) == static_cast<int>(k) + b(k));
    }
  };

  sctx.finalize();
}
