//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/__stf/places/exec/green_context.cuh>
#include <cuda/experimental/__stf/places/place_partition.cuh>
#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

// Green contexts are only supported since CUDA 12.4
#if _CCCL_CTK_AT_LEAST(12, 4)
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
#endif // _CCCL_CTK_AT_LEAST(12, 4)

int main()
{
#if _CCCL_CTK_BELOW(12, 4)
  fprintf(stderr, "Green contexts are not supported by this version of CUDA: skipping test.\n");
  return 0;
#else
  stream_ctx ctx;

  int NITER   = 8;
  const int n = 16 * 1024 * 1024;

  std::vector<double> X(n);
  std::vector<double> Y(n);

  for (int ind = 0; ind < n; ind++)
  {
    X[ind] = 1.0 * ind;
    Y[ind] = 2.0 * ind - 3.0;
  }

  auto handle_X = ctx.logical_data(make_slice(&X[0], n));
  auto handle_Y = ctx.logical_data(make_slice(&Y[0], n));

  auto where =
    exec_place::all_devices().partition_by_scope(ctx.async_resources(), place_partition_scope::green_context);

  comm_matrix_tracer tr;
  tr.init(where.size());

  for (int iter = 0; iter < NITER; iter++)
  {
    ctx.parallel_for(blocked_partition(), where, handle_X.shape(), handle_X.rw(), handle_Y.read())
        ->*[tr] __device__(size_t i, auto x, auto y) {
              x(i) += (y((i+n-1) % n) + y((i+n+1) % n))/2;
              tr.mark_access(pos4(i), shape(x), blocked_partition(), 1);
              tr.mark_access(pos4((i+n-1) % n), shape(y), blocked_partition(), 1);
              tr.mark_access(pos4((i+n+1) % n), shape(y), blocked_partition(), 1);
            };
  }

  ctx.finalize();

  tr.dump();
#endif // CUDA_VERSION < 12040
}
