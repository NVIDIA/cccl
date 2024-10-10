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

/*
 * In this example, the user provides streams in which the STF model inserts the proper dependencies
 */

static __global__ void cuda_sleep_kernel(long long int clock_cnt)
{
  long long int start_clock  = clock64();
  long long int clock_offset = 0;
  while (clock_offset < clock_cnt)
  {
    clock_offset = clock64() - start_clock;
  }
}

void cuda_sleep(double ms, cudaStream_t stream)
{
  int device;
  cudaGetDevice(&device);

  // cudaDevAttrClockRate: Peak clock frequency in kilohertz;
  int clock_rate;
  cudaDeviceGetAttribute(&clock_rate, cudaDevAttrClockRate, device);

  long long int clock_cnt = (long long int) (ms * clock_rate);
  cuda_sleep_kernel<<<1, 1, 0, stream>>>(clock_cnt);
}

int main()
{
  stream_ctx ctx;
  double vA, vB, vC, vD;
  auto A = ctx.logical_data(make_slice(&vA, 1));
  auto B = ctx.logical_data(make_slice(&vB, 1));
  auto C = ctx.logical_data(make_slice(&vC, 1));
  auto D = ctx.logical_data(make_slice(&vD, 1));

  // We are going to submit kernels with the following data accesses, where
  // K2 and K3 can be executed concurrently, after K1 and been executed, and
  // before K4 is executed.
  // K1(Aw); K2(Ar,Bw); K3(Ar, Cw); K4(Br,Cr,Dw);

  // User-provided streams
  cudaStream_t K1_stream;
  cudaStream_t K2_stream;
  cudaStream_t K3_stream;
  cudaStream_t K4_stream;
  cudaStreamCreate(&K1_stream);
  cudaStreamCreate(&K2_stream);
  cudaStreamCreate(&K3_stream);
  cudaStreamCreate(&K4_stream);

  // Kernel 1 : A(write)
  auto k1 = ctx.task(A.rw());
  k1.set_stream(K1_stream);
  k1.set_symbol("K1");
  k1.start();
  cuda_sleep(500, K1_stream);
  k1.end();

  // Kernel 2 : A(read) B(write)
  auto k2 = ctx.task(A.read(), B.write());
  k2.set_stream(K2_stream);
  k2.set_symbol("K2");
  k2.start();
  cuda_sleep(500, K2_stream);
  k2.end();

  // Kernel 3 : A(read) C(write)
  auto k3 = ctx.task(A.read(), C.write());
  k3.set_stream(K3_stream);
  k3.set_symbol("K3");
  k3.start();
  cuda_sleep(500, K3_stream);
  k3.end();

  // Kernel 4 : B(read) C(read) D(write)
  auto k4 = ctx.task(B.read(), C.read(), D.write());
  k4.set_stream(K4_stream);
  k4.set_symbol("K4");
  k4.start();
  cuda_sleep(500, K4_stream);
  k4.end();

  ctx.finalize();
}
