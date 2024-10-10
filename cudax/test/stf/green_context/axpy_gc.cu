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

void debug_info(cudaStream_t stream, CUgreenCtx g_ctx)
{
  // Get the green context associated to that CUDA stream
  CUgreenCtx stream_cugc;
  cuda_safe_call(cuStreamGetGreenCtx(CUstream(stream), &stream_cugc));
  assert(stream_cugc != nullptr);

  CUcontext stream_green_primary;
  CUcontext place_green_primary;

  unsigned long long stream_ctxId;
  unsigned long long place_ctxId;

  // Convert green contexts to primary contexts and get their ID
  cuda_safe_call(cuCtxFromGreenCtx(&stream_green_primary, stream_cugc));
  cuda_safe_call(cuCtxGetId(stream_green_primary, &stream_ctxId));

  cuda_safe_call(cuCtxFromGreenCtx(&place_green_primary, g_ctx));
  cuda_safe_call(cuCtxGetId(place_green_primary, &place_ctxId));

  // Make sure the stream belongs to the same green context as the execution place
  EXPECT(stream_ctxId == place_ctxId);
}
#endif // CUDA_VERSION >= 12040

int main()
{
#if CUDA_VERSION < 12040
  fprintf(stderr, "Green contexts are not supported by this version of CUDA: skipping test.\n");
  return 0;
#else
  int ndevs;
  const int num_sms = 8;
  cuda_safe_call(cudaGetDeviceCount(&ndevs));

  stream_ctx ctx;
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
  std::vector<green_context_helper> gc(ndevs);
  for (int devid = 0; devid < ndevs; devid++)
  {
    gc[devid] = green_context_helper(num_sms, devid);
  }

  for (int iter = 0; iter < NITER; iter++)
  {
    for (int devid = 0; devid < ndevs; devid++)
    {
      auto& g_ctx = gc[devid];
      auto cnt    = g_ctx.get_count();
      ctx.task(exec_place::green_ctx(g_ctx.get_view(iter % cnt)), handle_X.read(), handle_Y.rw())
          ->*[&](cudaStream_t stream, auto dX, auto dY) {
                debug_info(stream, g_ctx.get_view(iter % cnt).g_ctx);
                axpy<<<16, 16, 0, stream>>>(alpha, dX, dY);
              };
    }
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
