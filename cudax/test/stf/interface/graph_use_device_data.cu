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
// #include <cuda/experimental/__stf/graph/interfaces/slice.cuh>

using namespace cuda::experimental::stf;

template <typename T>
__global__ void axpy(int N, T a, T* x, T* y)
{
  int tid      = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;

  for (int ind = tid; ind < N; ind += nthreads)
  {
    y[ind] += a * x[ind];
  }
}

template <typename T>
__global__ void setup_vectors(int N, T* x, T* y, T* z)
{
  int tid      = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;

  for (int ind = tid; ind < N; ind += nthreads)
  {
    x[ind] = 1.0 * ind;
    y[ind] = 2.0 * ind - 3.0;
    z[ind] = 7.0 * ind + 6.0;
  }
}

int main(int argc, char** argv)
{
  graph_ctx ctx;
  const int N        = 12;
  const double alpha = 2.0;

  double *dX, *dY, *dZ;

  cuda_safe_call(cudaMalloc((void**) &dX, N * sizeof(double)));
  cuda_safe_call(cudaMalloc((void**) &dY, N * sizeof(double)));
  cuda_safe_call(cudaMalloc((void**) &dZ, N * sizeof(double)));

  // Use a kernel to setup values
  setup_vectors<<<16, 16>>>(N, dX, dY, dZ);
  cuda_safe_call(cudaDeviceSynchronize());

  // We here provide device addresses and memory node 1 (which is assumed to
  // be device 0)
  auto handle_X = ctx.logical_data(make_slice(dX, N), data_place::device(0));
  auto handle_Y = ctx.logical_data(make_slice(dY, N), data_place::device(0));
  auto handle_Z = ctx.logical_data(make_slice(dZ, N), data_place::device(0));

  // Y = Y + alpha X
  ctx.task(handle_X.read(), handle_Y.rw())->*[&](cudaStream_t stream, auto dX, auto dY) {
    axpy<<<16, 16, 0, stream>>>(N, alpha, dX.data_handle(), dY.data_handle());
  };

  // Z = Z + alpha X
  ctx.task(handle_X.read(), handle_Z.rw())->*[&](cudaStream_t stream, auto dX, auto dZ) {
    axpy<<<16, 16, 0, stream>>>(N, alpha, dX.data_handle(), dZ.data_handle());
  };

  // Z = Z + alpha Y
  ctx.task(handle_Y.read(), handle_Z.rw())->*[&](cudaStream_t stream, auto dY, auto dZ) {
    axpy<<<16, 16, 0, stream>>>(N, alpha, dY.data_handle(), dZ.data_handle());
  };

  ctx.host_launch(handle_X.read(), handle_Y.read(), handle_Z.read())->*[&](auto hX, auto hY, auto hZ) {
    for (size_t ind = 0; ind < N; ind++)
    {
      // X unchanged
      EXPECT(fabs(hX(ind) - 1.0 * ind) < 0.00001);
      // Y = Y + alpha X
      EXPECT(fabs(hY(ind) - (-3.0 + ind * (2.0 + alpha))) < 0.00001);
      // Z = Z + alpha (X + alpha Y)
      EXPECT(fabs(hZ(ind) - ((6.0 - 3 * alpha) + ind * (7.0 + 3 * alpha + alpha * alpha))) < 0.00001);
    }
  };

  ctx.submit();

  if (argc > 1)
  {
    std::cout << "Generating DOT output in " << argv[1] << std::endl;
    ctx.print_to_dot(argv[1]);
  }

  ctx.finalize();
}
