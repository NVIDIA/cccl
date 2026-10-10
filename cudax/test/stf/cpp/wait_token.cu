//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

//! \file
//!
//! \brief Test ctx.wait() on a token: a blocking, value-less synchronization

#include <cuda/experimental/stf.cuh>

#include <type_traits>

using namespace cuda::experimental::stf;

__global__ void set_value(int* p, int v)
{
  *p = v;
}

template <typename context_t>
void run()
{
  context_t ctx;

  // Externally owned buffer: STF only schedules around it, it never owns it.
  int* d_val = nullptr;
  cuda_safe_call(cudaMalloc(&d_val, sizeof(int)));

  auto tok = ctx.token();

  ctx.task(tok.write())->*[=](cudaStream_t s) {
    set_value<<<1, 1, 0, s>>>(d_val, 42);
  };

  // wait(token) has no value to materialize: it must return void and only
  // block the host until the token's producing work has completed.
  static_assert(::std::is_void_v<decltype(ctx.wait(tok))>, "wait(token) must return void");
  ctx.wait(tok);

  int h_val = 0;
  cuda_safe_call(cudaMemcpy(&h_val, d_val, sizeof(int), cudaMemcpyDeviceToHost));
  _CCCL_ASSERT(h_val == 42, "wait(token) did not synchronize the producing task");

  ctx.finalize();
  cuda_safe_call(cudaFree(d_val));
}

int main()
{
  run<stream_ctx>();
  run<graph_ctx>();
  run<context>();
  run<stackable_ctx>();
}
