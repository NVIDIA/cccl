//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Ensure we can use the same logical data multiple time in a task
 */

#include <cuda/experimental/__stf/graph/graph_ctx.cuh>
#include <cuda/experimental/__stf/stream/stream_ctx.cuh>

using namespace cuda::experimental::stf;

template <typename T>
__global__ void diff_cnt(int n, T* x, T* y, int* delta)
{
  int tid      = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;

  for (int ind = tid; ind < n; ind += nthreads)
  {
    if (y[ind] != x[ind])
    {
      atomicAdd(delta, 1);
    }
  }
}

template <typename Ctx, typename T>
void compare_two_vectors(Ctx& ctx, logical_data<T>& a, logical_data<T>& b, int& delta)
{
  auto delta_cnt = ctx.logical_data(make_slice(&delta, 1));
  const auto n   = a.shape().extent(0);

  // Count the number of differences
  ctx.task(a.read(), b.read(), delta_cnt.rw())->*[=](cudaStream_t stream, auto da, auto db, auto ddelta) {
    diff_cnt<<<16, 128, 0, stream>>>(static_cast<int>(n), da.data_handle(), db.data_handle(), ddelta.data_handle());
  };

  // Read that value on the host
  ctx.host_launch(delta_cnt.read())->*[&](auto /*unused*/) {};
}

static const size_t N = 12;

template <class Ctx>
void run(double (&X)[N], double (&Y)[N])
{
  Ctx ctx;
  auto handle_X = ctx.logical_data(X);
  auto handle_Y = ctx.logical_data(Y);

  int ret1 = 0, ret2 = 0;

  compare_two_vectors(ctx, handle_X, handle_Y, ret1);
  compare_two_vectors(ctx, handle_X, handle_X, ret2);

  ctx.finalize();

  // After sync, we can inspect the returned values.
  // First two vectors are different
  assert(ret1 > 0);
  // Other two vectors are equal
  assert(ret2 == 0);
}

int main()
{
  double X[N], Y[N];

  for (size_t ind = 0; ind < N; ind++)
  {
    X[ind] = 1.0 * ind;
    Y[ind] = 2.0 * ind - 3.0;
  }

  run<stream_ctx>(X, Y);
  run<graph_ctx>(X, Y);
}
