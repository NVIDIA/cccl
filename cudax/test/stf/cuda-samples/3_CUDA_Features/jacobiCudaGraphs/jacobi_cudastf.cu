//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// Jacobi method on a linear system A*x = b,
// where A is diagonally dominant and the exact solution consists
// of all ones.

#include <cuda/experimental/__stf/stream/stream_ctx.cuh>

#define N_ROWS 512

namespace cg = cooperative_groups;

using namespace cuda::experimental::stf;

// 8 Rows of square-matrix A processed by each CTA.
// This can be max 32 and only power of 2 (i.e., 2/4/8/16/32).
#define ROWS_PER_CTA 8

// creates N_ROWS x N_ROWS matrix A with N_ROWS+1 on the diagonal and 1
// elsewhere. The elements of the right hand side b all equal 2*n, hence the
// exact solution x to A*x = b is a vector of ones.
void createLinearSystem(float* A, double* b)
{
  int i, j;
  for (i = 0; i < N_ROWS; i++)
  {
    b[i] = 2.0 * N_ROWS;
    for (j = 0; j < N_ROWS; j++)
    {
      A[i * N_ROWS + j] = 1.0;
    }
    A[i * N_ROWS + i] = N_ROWS + 1.0;
  }
}

static __global__ void
JacobiMethod(const float* A, const double* b, const float conv_threshold, double* x, double* x_new, double* sum)
{
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  __shared__ double x_shared[N_ROWS]; // N_ROWS == n
  __shared__ double b_shared[ROWS_PER_CTA + 1];

  for (int i = threadIdx.x; i < N_ROWS; i += blockDim.x)
  {
    x_shared[i] = x[i];
  }

  if (threadIdx.x < ROWS_PER_CTA)
  {
    int k = threadIdx.x;
    for (int i = k + (blockIdx.x * ROWS_PER_CTA); (k < ROWS_PER_CTA) && (i < N_ROWS);
         k += ROWS_PER_CTA, i += ROWS_PER_CTA)
    {
      b_shared[i % (ROWS_PER_CTA + 1)] = b[i];
    }
  }

  cg::sync(cta);

  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  for (int k = 0, i = blockIdx.x * ROWS_PER_CTA; (k < ROWS_PER_CTA) && (i < N_ROWS); k++, i++)
  {
    double rowThreadSum = 0.0;
    for (int j = threadIdx.x; j < N_ROWS; j += blockDim.x)
    {
      rowThreadSum += (A[i * N_ROWS + j] * x_shared[j]);
    }

    for (int offset = tile32.size() / 2; offset > 0; offset /= 2)
    {
      rowThreadSum += tile32.shfl_down(rowThreadSum, offset);
    }

    if (tile32.thread_rank() == 0)
    {
      atomicAdd(&b_shared[i % (ROWS_PER_CTA + 1)], -rowThreadSum);
    }
  }

  cg::sync(cta);

  if (threadIdx.x < ROWS_PER_CTA)
  {
    cg::thread_block_tile<ROWS_PER_CTA> tile8 = cg::tiled_partition<ROWS_PER_CTA>(cta);
    double temp_sum                           = 0.0;

    int k = threadIdx.x;

    for (int i = k + (blockIdx.x * ROWS_PER_CTA); (k < ROWS_PER_CTA) && (i < N_ROWS);
         k += ROWS_PER_CTA, i += ROWS_PER_CTA)
    {
      double dx = b_shared[i % (ROWS_PER_CTA + 1)];
      dx /= A[i * N_ROWS + i];

      x_new[i] = (x_shared[i] + dx);
      temp_sum += fabs(dx);
    }

    for (int offset = tile8.size() / 2; offset > 0; offset /= 2)
    {
      temp_sum += tile8.shfl_down(temp_sum, offset);
    }

    if (tile8.thread_rank() == 0)
    {
      atomicAdd(sum, temp_sum);
    }
  }
}

// Thread block size for finalError kernel should be multiple of 32
static __global__ void finalError(const double* x, double* g_sum)
{
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  extern __shared__ double warpSum[];
  double sum = 0.0;

  int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = globalThreadId; i < N_ROWS; i += blockDim.x * gridDim.x)
  {
    double d = x[i] - 1.0;
    sum += fabs(d);
  }

  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  for (int offset = tile32.size() / 2; offset > 0; offset /= 2)
  {
    sum += tile32.shfl_down(sum, offset);
  }

  if (tile32.thread_rank() == 0)
  {
    warpSum[threadIdx.x / warpSize] = sum;
  }

  cg::sync(cta);

  double blockSum = 0.0;
  if (threadIdx.x < (blockDim.x / warpSize))
  {
    blockSum = warpSum[threadIdx.x];
  }

  if (threadIdx.x < 32)
  {
    for (int offset = tile32.size() / 2; offset > 0; offset /= 2)
    {
      blockSum += tile32.shfl_down(blockSum, offset);
    }
    if (tile32.thread_rank() == 0)
    {
      atomicAdd(g_sum, blockSum);
    }
  }
}

// Run the Jacobi method for A*x = b on GPU without CUDA Graph.
// double JacobiMethodGpu(const float *A, const double *b,
//                       const float conv_threshold, const int max_iter,
//                       double *x, double *x_new) {
double JacobiMethodGpu(
  stream_ctx& ctx,
  logical_data<slice<float>>& A_handle,
  logical_data<slice<double>>& b_handle,
  const float conv_threshold,
  const int max_iter,
  logical_data<slice<double>>& x_handle,
  logical_data<slice<double>>& x_new_handle)
{
  // CTA size
  dim3 nthreads(256, 1, 1);
  // grid size
  dim3 nblocks((N_ROWS / ROWS_PER_CTA) + 2, 1, 1);

  double sum = 0.0;

  // Task for the whole Jacobi
  auto sum_handle = ctx.logical_data(&sum, 1).set_symbol("sum");

  int k;
  for (k = 0; k < max_iter; k++)
  {
    auto x_mode     = (k & 1) == 0 ? access_mode::read : access_mode::rw;
    auto x_new_mode = (k & 1) == 0 ? access_mode::rw : access_mode::read;

    ctx.task(A_handle.read(),
             b_handle.read(),
             task_dep<slice<double>>(x_handle, x_mode),
             task_dep<slice<double>>(x_new_handle, x_new_mode),
             sum_handle.write())
        .set_symbol("JacobiMethod")
        ->*
      [&](cudaStream_t stream, auto A, auto b, auto x, auto x_new, auto d_sum) {
        cuda_try(cudaMemsetAsync(d_sum.data_handle(), 0, sizeof(double), stream));

        if ((k & 1) == 0)
        {
          JacobiMethod<<<nblocks, nthreads, 0, stream>>>(
            A.data_handle(), b.data_handle(), conv_threshold, x.data_handle(), x_new.data_handle(), d_sum.data_handle());
        }
        else
        {
          JacobiMethod<<<nblocks, nthreads, 0, stream>>>(
            A.data_handle(), b.data_handle(), conv_threshold, x_new.data_handle(), x.data_handle(), d_sum.data_handle());
        }
      };

    bool converged;
    // Check if sum is smaller than the threshold and halt the loop if this is true
    auto t_host = ctx.task(sum_handle.read());
    t_host.on(exec_place::host);
    t_host.set_symbol("check_converged");
    t_host->*[&](cudaStream_t stream, auto sum2) {
      cuda_try(cudaStreamSynchronize(stream));
      // fprintf(stderr, "SUM %e\n", sum);
      assert(*sum2.data_handle() == sum);
      converged = (*sum2.data_handle() <= conv_threshold);
    };

    // read sum
    if (converged)
    {
      break;
    }
  }

  auto final_x_handle = ((k & 1) == 0) ? &x_new_handle : &x_handle;
  auto t              = ctx.task(sum_handle.write(), final_x_handle->read());
  t.set_symbol("finalError");
  t->*[&](cudaStream_t stream, auto d_sum, auto final_x) {
    cuda_try(cudaMemsetAsync(d_sum.data_handle(), 0, sizeof(double), stream));

    nblocks.x            = (N_ROWS / nthreads.x) + 1;
    size_t sharedMemSize = ((nthreads.x / 32) + 1) * sizeof(double);
    finalError<<<nblocks, nthreads, sharedMemSize, stream>>>(final_x.data_handle(), d_sum.data_handle());
  };

  auto t_display = ctx.task(sum_handle.read());
  t_display.on(exec_place::host);
  t_display.set_symbol("finalError");
  t_display->*[&](cudaStream_t stream, auto /*unused*/) {
    cuda_try(cudaStreamSynchronize(stream));
    // printf("GPU iterations : %d\n", k + 1);
    // printf("GPU error: %.3e\n", *h_sum.data_handle());
  };

  return sum;
}

// Run the Jacobi method for A*x = b on CPU.
void JacobiMethodCPU(float* A, double* b, float conv_threshold, int max_iter, int* num_iter, double* x)
{
  double* x_new = (double*) calloc(N_ROWS, sizeof(double));
  SCOPE(exit)
  {
    free(x_new);
  };

  int k = 0;

  for (; k < max_iter; k++)
  {
    double sum = 0.0;
    for (int i = 0; i < N_ROWS; i++)
    {
      double temp_dx = b[i];
      for (int j = 0; j < N_ROWS; j++)
      {
        temp_dx -= A[i * N_ROWS + j] * x[j];
      }
      temp_dx /= A[i * N_ROWS + i];
      x_new[i] += temp_dx;
      sum += fabs(temp_dx);
    }

    for (int i = 0; i < N_ROWS; i++)
    {
      x[i] = x_new[i];
    }

    if (sum <= conv_threshold)
    {
      break;
    }
  }

  *num_iter = k + 1;
}

template <typename Ctx>
int run()
{
  Ctx ctx;

  double* b = cuda_try<cudaMallocHost<double>>(N_ROWS * sizeof(double), 0);
  SCOPE(exit)
  {
    cuda_try(cudaFreeHost(b));
  };

  float* A = cuda_try<cudaMallocHost<float>>(N_ROWS * N_ROWS * sizeof(float), 0);
  SCOPE(exit)
  {
    cuda_try(cudaFreeHost(A));
  };

  memset(b, 0, N_ROWS * sizeof(double));

  memset(A, 0, N_ROWS * N_ROWS * sizeof(float));

  createLinearSystem(A, b);
  // start with array of all zeroes
  double* x = (double*) calloc(N_ROWS, sizeof(double));
  SCOPE(exit)
  {
    free(x);
  };

  auto A_handle     = ctx.logical_data(A, N_ROWS * N_ROWS);
  auto b_handle     = ctx.logical_data(b, N_ROWS);
  auto x_handle     = ctx.logical_data(x, N_ROWS);
  auto x_new_handle = ctx.template logical_data<double>(N_ROWS);

  A_handle.set_symbol("A");
  b_handle.set_symbol("b");
  x_handle.set_symbol("x");
  x_new_handle.set_symbol("x_new");

  float conv_threshold = 1.0e-2;
  int max_iter         = 4 * N_ROWS * N_ROWS;
  int cnt              = 0;

  JacobiMethodCPU(A, b, conv_threshold, max_iter, &cnt, x);

  double sum = 0.0;
  // Compute error
  for (int i = 0; i < N_ROWS; i++)
  {
    double d = x[i] - 1.0;
    sum += fabs(d);
  }

  ctx.task(x_handle.write()).set_symbol("memset x")->*[&](cudaStream_t stream, auto d_x) {
    cuda_try(cudaMemsetAsync(d_x.data_handle(), 0, sizeof(double) * N_ROWS, stream));
  };

  ctx.task(x_new_handle.write()).set_symbol("memset x_new")->*[](cudaStream_t stream, auto d_x_new) {
    cuda_try(cudaMemsetAsync(d_x_new.data_handle(), 0, sizeof(double) * N_ROWS, stream));
  };

  double sumGPU = JacobiMethodGpu(ctx, A_handle, b_handle, conv_threshold, max_iter, x_handle, x_new_handle);

  ctx.finalize();

  if (fabs(sum - sumGPU) > conv_threshold)
  {
    printf("&&&& jacobiCudaGraphs FAILED\n");
    return EXIT_FAILURE;
  }
  return 0;
}

int main()
{
  return run<stream_ctx>();
  // run<graph_ctx>();
}
