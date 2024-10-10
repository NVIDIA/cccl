//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/__stf/places/exec/green_context.cuh>
#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

// Green contexts are only supported since CUDA 12.4
#if CUDA_VERSION >= 12040
__global__ void axpy(double a, slice<const double> x, slice<double> y)
{
  int tid      = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;

  size_t n = x.extent(0);
  for (int ind = tid; ind < n; ind += nthreads)
  {
    y(ind) += a * x(ind);
  }
}
#endif // CUDA_VERSION >= 12040

int main()
{
#if CUDA_VERSION < 12040
  fprintf(stderr, "Green contexts are not supported by this version of CUDA: skipping test.\n");
  return 0;
#else
  int ndevs;
  const int num_sms = 16;
  cuda_safe_call(cudaGetDeviceCount(&ndevs));

  graph_ctx ctx;
  const double alpha = 2.0;

  int NITER   = 30;
  const int n = 12;

  double X[n], Y[n];

  for (int ind = 0; ind < n; ind++)
  {
    X[ind] = 1.0 * ind;
    Y[ind] = 2.0 * ind - 3.0;
  }

  auto handle_X = ctx.logical_data(make_slice(&X[0], n));
  auto handle_Y = ctx.logical_data(make_slice(&Y[0], n));

  // The green_context_helper class automates the creation of green context views
  green_context_helper gc(num_sms);

  for (int iter = 0; iter < NITER; iter++)
  {
    auto cnt = gc.get_count();
    ctx.task(exec_place::green_ctx(gc.get_view(iter % cnt)), handle_X.read(), handle_Y.rw())
        ->*[&](cudaStream_t stream, auto dX, auto dY) {
              axpy<<<16, 16, 0, stream>>>(alpha, dX, dY);
            };
  }

  ctx.host_launch(handle_X.read(), handle_Y.read())->*[&](auto hX, auto hY) {
    for (int ind = 0; ind < n; ind++)
    {
      EXPECT(fabs(hX(ind) - 1.0 * ind) < 0.00001);
      EXPECT(fabs(hY(ind) - (2.0 * ind - 3.0) - NITER * ndevs * alpha * hX(ind)) < 0.00001);
    }
  };

  ctx.finalize();
#endif // CUDA_VERSION < 12040
}
