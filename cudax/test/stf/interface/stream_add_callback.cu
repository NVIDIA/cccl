//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/__stf/stream/stream_ctx.cuh>

using namespace cuda::experimental::stf;

void host_inc(cudaStream_t /*unused*/, cudaError_t /*unused*/, void* userData)
{
  /* Retrieve a pointer to the arguments and destroy it */
  int* var = static_cast<int*>(userData);
  *var     = *var + 1;
}

int main()
{
  int cnt = 0;

  stream_ctx ctx;
  auto h_cnt = ctx.logical_data(make_slice(&cnt, 1));

  int NITER = 2;
  for (int iter = 0; iter < NITER; iter++)
  {
    // Enqueue a dummy GPU task
    ctx.task(h_cnt.rw())->*[&](cudaStream_t /*unused*/, auto /*unused*/) {
      // no-op
    };

    // Enqueue a host callback
    ctx.task(exec_place::host, h_cnt.rw())->*[&](cudaStream_t stream, auto s_cnt) {
      cuda_safe_call(cudaStreamAddCallback(stream, host_inc, s_cnt.data_handle(), 0));
      cuda_safe_call(cudaGetLastError());
    };
  }

  // Ask to use Y on the host
  ctx.host_launch(h_cnt.read())->*[&](auto s_cnt) {
    EXPECT(s_cnt(0) == NITER);
  };

  ctx.finalize();
}
