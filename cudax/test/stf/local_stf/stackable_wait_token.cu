//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief Ensure ctx.wait() works on a token in a (root) stackable context
 */

#include <cuda/experimental/stf.cuh>

#include <type_traits>

using namespace cuda::experimental::stf;

int main()
{
  stackable_ctx sctx;

  // Externally owned buffer: the stackable context only schedules around it.
  int* d_val = nullptr;
  cuda_safe_call(cudaMalloc(&d_val, sizeof(int)));

  auto tok = sctx.token();

  sctx.parallel_for(box(1), tok.rw())->*[d_val] __device__(size_t) {
    *d_val = 42;
  };

  // wait(token) forwards to the underlying context and returns void: it only
  // blocks the host until the token's producing work has completed.
  static_assert(::std::is_void_v<decltype(sctx.wait(tok))>, "wait(token) must return void");
  sctx.wait(tok);

  int h_val = 0;
  cuda_safe_call(cudaMemcpy(&h_val, d_val, sizeof(int), cudaMemcpyDeviceToHost));
  _CCCL_ASSERT(h_val == 42, "wait(token) did not synchronize the producing task");

  sctx.finalize();
  cuda_safe_call(cudaFree(d_val));
}
