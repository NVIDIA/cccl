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

template <typename T>
__global__ void axpy(int n, T a, T* x, T* y)
{
  int tid      = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;

  for (int ind = tid; ind < n; ind += nthreads)
  {
    y[ind] += a * x[ind];
  }
}

template <typename T>
__global__ void setup_vectors(int n, T* x, T* y, T* z)
{
  int tid      = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;

  for (int ind = tid; ind < n; ind += nthreads)
  {
    x[ind] = 1.0 * ind;
    y[ind] = 2.0 * ind - 3.0;
    z[ind] = 7.0 * ind + 6.0;
  }
}

int main()
{
// FIXME
#if 0
    stream_ctx ctx;

    cudaStream_t stream;
    cuda_safe_call(cudaStreamCreate(&stream));

    const double alpha = 2.0;

    size_t n = 12;

    double *dX, *dY, *dZ;
    cuda_safe_call(cudaMallocAsync((void**) &dX, n * sizeof(double), stream));
    cuda_safe_call(cudaMallocAsync((void**) &dY, n * sizeof(double), stream));
    cuda_safe_call(cudaMallocAsync((void**) &dZ, n * sizeof(double), stream));

    // Use a kernel to setup values
    setup_vectors<<<16, 16, 0, stream>>>(n, dX, dY, dZ);

    auto stream_ready = make_event(stream);

    // We here provide device addresses and memory node 1 (which is assumed to
    // be device 0)
    auto handle_X = ctx.logical_data(slice<double>(dX, { n }), data_place::device(0), std::string("X"), stream_ready);
    auto handle_Y = ctx.logical_data(slice<double>(dY, { n }), data_place::device(0), std::string("Y"), stream_ready);
    auto handle_Z = ctx.logical_data(slice<double>(dZ, { n }), data_place::device(0), std::string("Z"), stream_ready);

    ctx.task(handle_X.read(), handle_Y.rw())->*[](cudaStream_t stream, auto X, auto Y) {
        axpy<<<16, 128, 0, stream>>>(n, alpha, X.base(), Y.base());
    };

    ctx.task(handle_X.read(), handle_Z.rw())->*[](cudaStream_t stream, auto X, auto Z) {
        axpy<<<16, 128, 0, stream>>>(n, alpha, X.base(), Z.base());
    };

    ctx.task(handle_Y.read(), handle_Z.rw())->*[](cudaStream_t stream, auto Y, auto Z) {
        axpy<<<16, 128, 0, stream>>>(n, alpha, Y.base(), Z.base());
    };

    // Access Ask to use X, Y and Z on the host
    ctx.task(exec_place::host, handle_X.read(), handle_Y.read(), handle_Z.read())
                    ->*
            [](cudaStream_t stream, auto X, auto Y, auto Z) {
                cuda_safe_call(cudaStreamSynchronize(stream));

                for (int ind = 0; ind < n; ind++) {
                    // X unchanged
                    assert(fabs(X(ind) - 1.0 * ind) < 0.00001);
                    // Y = Y + alpha X
                    assert(fabs(Y(ind) - (-3.0 + ind * (2.0 + alpha))) < 0.00001);
                    // Z = Z + alpha (X + alpha Y)
                    assert(fabs(Z(ind) - ((6.0 - 3 * alpha) + ind * (7.0 + 3 * alpha + alpha * alpha))) < 0.00001);
                }
            };

    ctx.finalize();
#endif

  return 0;
}
