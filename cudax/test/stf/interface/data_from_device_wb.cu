//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/__stf/graph/graph_ctx.cuh>
#include <cuda/experimental/__stf/stream/stream_ctx.cuh>

using namespace cuda::experimental::stf;

/*
 * This test makes sure write-back works even if the original data place was a device
 */

template <typename T>
__global__ void setup(slice<T> s)
{
  int tid      = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;

  for (int ind = tid; ind < s.size(); ind += nthreads)
  {
    s(ind) = 1.0 * ind;
  }
}

template <typename T>
__global__ void check(T* x, size_t n)
{
  int tid      = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;

  for (int ind = tid; ind < n; ind += nthreads)
  {
    assert(x[ind] == 2.0 * ind + 1.0);
  }
}

template <typename Ctx>
void run()
{
  Ctx ctx;
  const size_t n = 12;

  double* dX;
  cuda_safe_call(cudaMalloc((void**) &dX, n * sizeof(double)));

  // We here provide device addresses and memory node 1 (which is assumed to
  // be device 0)
  auto handle_X = ctx.logical_data(make_slice(dX, n), data_place::device(0));

  ctx.task(handle_X.write())->*[&](cudaStream_t stream, auto X) {
    setup<<<16, 128, 0, stream>>>(X);
  };

  ctx.host_launch(handle_X.rw())->*[&](auto X) {
    for (size_t ind = 0; ind < n; ind++)
    {
      X(ind) = 2.0 * X(ind) + 1.0;
    }
  };

  ctx.finalize();

  // Check if data was properly written-back with a blocking kernel
  check<<<16, 128>>>(dX, n);
}

int main()
{
  run<stream_ctx>();
  run<graph_ctx>();
}
