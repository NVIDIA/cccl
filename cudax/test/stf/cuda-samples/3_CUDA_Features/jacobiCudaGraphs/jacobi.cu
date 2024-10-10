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

// This sample demonstrates Instantiated CUDA Graph Update
// with Jacobi Iterative Method in 3 different methods:
// 1 - JacobiMethodGpuCudaGraphExecKernelSetParams() - CUDA Graph with
// cudaGraphExecKernelNodeSetParams() 2 - JacobiMethodGpuCudaGraphExecUpdate() -
// CUDA Graph with cudaGraphExecUpdate() 3 - JacobiMethodGpu() - Non CUDA Graph
// method

// Jacobi method on a linear system A*x = b,
// where A is diagonally dominant and the exact solution consists
// of all ones.

#include <cuda/experimental/__stf/utility/cuda_safe_call.cuh>

#include <cooperative_groups.h>

using cuda::experimental::stf::cuda_safe_call;

#define N_ROWS 512

namespace cg = cooperative_groups;

// 8 Rows of square-matrix A processed by each CTA.
// This can be max 32 and only power of 2 (i.e., 2/4/8/16/32).
#define ROWS_PER_CTA 8

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
  unsigned long long int* address_as_ull = (unsigned long long int*) address;
  unsigned long long int old             = *address_as_ull, assumed;

  do
  {
    assumed = old;
    old     = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN !=
    // NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}
#endif

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
static __global__ void finalError(double* x, double* g_sum)
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

// Run the Jacobi method for A*x = b on GPU with CUDA Graph -
// cudaGraphExecKernelNodeSetParams().
double JacobiMethodGpuCudaGraphExecKernelSetParams(
  const float* A,
  const double* b,
  const float conv_threshold,
  const int max_iter,
  double* x,
  double* x_new,
  cudaStream_t stream)
{
  // CTA size
  dim3 nthreads(256, 1, 1);
  // grid size
  dim3 nblocks((N_ROWS / ROWS_PER_CTA) + 2, 1, 1);
  cudaGraph_t graph;
  cudaGraphExec_t graphExec = NULL;

  double sum    = 0.0;
  double* d_sum = NULL;
  cuda_safe_call(cudaMalloc(&d_sum, sizeof(double)));

  std::vector<cudaGraphNode_t> nodeDependencies;
  cudaGraphNode_t memcpyNode, jacobiKernelNode, memsetNode;
  cudaMemcpy3DParms memcpyParams;
  cudaMemsetParams memsetParams;

  memsetParams.dst   = (void*) d_sum;
  memsetParams.value = 0;
  memsetParams.pitch = 0;
  // elementSize can be max 4 bytes, so we take sizeof(float) and width=2
  memsetParams.elementSize = sizeof(float);
  memsetParams.width       = 2;
  memsetParams.height      = 1;

  cuda_safe_call(cudaGraphCreate(&graph, 0));
  cuda_safe_call(cudaGraphAddMemsetNode(&memsetNode, graph, NULL, 0, &memsetParams));
  nodeDependencies.push_back(memsetNode);

  cudaKernelNodeParams NodeParams0, NodeParams1;
  NodeParams0.func           = (void*) JacobiMethod;
  NodeParams0.gridDim        = nblocks;
  NodeParams0.blockDim       = nthreads;
  NodeParams0.sharedMemBytes = 0;
  void* kernelArgs0[6] = {(void*) &A, (void*) &b, (void*) &conv_threshold, (void*) &x, (void*) &x_new, (void*) &d_sum};
  NodeParams0.kernelParams = kernelArgs0;
  NodeParams0.extra        = NULL;

  cuda_safe_call(
    cudaGraphAddKernelNode(&jacobiKernelNode, graph, nodeDependencies.data(), nodeDependencies.size(), &NodeParams0));

  nodeDependencies.clear();
  nodeDependencies.push_back(jacobiKernelNode);

  memcpyParams.srcArray = NULL;
  memcpyParams.srcPos   = make_cudaPos(0, 0, 0);
  memcpyParams.srcPtr   = make_cudaPitchedPtr(d_sum, sizeof(double), 1, 1);
  memcpyParams.dstArray = NULL;
  memcpyParams.dstPos   = make_cudaPos(0, 0, 0);
  memcpyParams.dstPtr   = make_cudaPitchedPtr(&sum, sizeof(double), 1, 1);
  memcpyParams.extent   = make_cudaExtent(sizeof(double), 1, 1);
  memcpyParams.kind     = cudaMemcpyDeviceToHost;

  cuda_safe_call(
    cudaGraphAddMemcpyNode(&memcpyNode, graph, nodeDependencies.data(), nodeDependencies.size(), &memcpyParams));

  cuda_safe_call(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

  NodeParams1.func           = (void*) JacobiMethod;
  NodeParams1.gridDim        = nblocks;
  NodeParams1.blockDim       = nthreads;
  NodeParams1.sharedMemBytes = 0;
  void* kernelArgs1[6] = {(void*) &A, (void*) &b, (void*) &conv_threshold, (void*) &x_new, (void*) &x, (void*) &d_sum};
  NodeParams1.kernelParams = kernelArgs1;
  NodeParams1.extra        = NULL;

  int k = 0;
  for (k = 0; k < max_iter; k++)
  {
    cuda_safe_call(
      cudaGraphExecKernelNodeSetParams(graphExec, jacobiKernelNode, ((k & 1) == 0) ? &NodeParams0 : &NodeParams1));
    cuda_safe_call(cudaGraphLaunch(graphExec, stream));
    cuda_safe_call(cudaStreamSynchronize(stream));

    if (sum <= conv_threshold)
    {
      cuda_safe_call(cudaMemsetAsync(d_sum, 0, sizeof(double), stream));
      nblocks.x            = (N_ROWS / nthreads.x) + 1;
      size_t sharedMemSize = ((nthreads.x / 32) + 1) * sizeof(double);
      if ((k & 1) == 0)
      {
        finalError<<<nblocks, nthreads, sharedMemSize, stream>>>(x_new, d_sum);
      }
      else
      {
        finalError<<<nblocks, nthreads, sharedMemSize, stream>>>(x, d_sum);
      }

      cuda_safe_call(cudaMemcpyAsync(&sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost, stream));
      cuda_safe_call(cudaStreamSynchronize(stream));
      // printf("GPU iterations : %d\n", k + 1);
      // printf("GPU error: %.3e\n", sum);
      break;
    }
  }

  cuda_safe_call(cudaFree(d_sum));
  return sum;
}

// Run the Jacobi method for A*x = b on GPU with Instantiated CUDA Graph Update
// API - cudaGraphExecUpdate().
double JacobiMethodGpuCudaGraphExecUpdate(
  const float* A,
  const double* b,
  const float conv_threshold,
  const int max_iter,
  double* x,
  double* x_new,
  cudaStream_t stream)
{
  // CTA size
  dim3 nthreads(256, 1, 1);
  // grid size
  dim3 nblocks((N_ROWS / ROWS_PER_CTA) + 2, 1, 1);
  cudaGraph_t graph;
  cudaGraphExec_t graphExec = NULL;

  double sum = 0.0;
  double* d_sum;
  cuda_safe_call(cudaMalloc(&d_sum, sizeof(double)));

  int k = 0;
  for (k = 0; k < max_iter; k++)
  {
    cuda_safe_call(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    cuda_safe_call(cudaMemsetAsync(d_sum, 0, sizeof(double), stream));
    if ((k & 1) == 0)
    {
      JacobiMethod<<<nblocks, nthreads, 0, stream>>>(A, b, conv_threshold, x, x_new, d_sum);
    }
    else
    {
      JacobiMethod<<<nblocks, nthreads, 0, stream>>>(A, b, conv_threshold, x_new, x, d_sum);
    }
    cuda_safe_call(cudaMemcpyAsync(&sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost, stream));
    cuda_safe_call(cudaStreamEndCapture(stream, &graph));

    if (graphExec == NULL)
    {
      cuda_safe_call(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
    }
    else
    {
      cudaGraphExecUpdateResult updateResult_out;
      cuda_safe_call(cudaGraphExecUpdate(graphExec, graph, NULL, &updateResult_out));
      if (updateResult_out != cudaGraphExecUpdateSuccess)
      {
        if (graphExec != NULL)
        {
          cuda_safe_call(cudaGraphExecDestroy(graphExec));
        }
        printf("k = %d graph update failed with error - %d\n", k, updateResult_out);
        cuda_safe_call(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
      }
    }
    cuda_safe_call(cudaGraphLaunch(graphExec, stream));
    cuda_safe_call(cudaStreamSynchronize(stream));

    if (sum <= conv_threshold)
    {
      cuda_safe_call(cudaMemsetAsync(d_sum, 0, sizeof(double), stream));
      nblocks.x            = (N_ROWS / nthreads.x) + 1;
      size_t sharedMemSize = ((nthreads.x / 32) + 1) * sizeof(double);
      if ((k & 1) == 0)
      {
        finalError<<<nblocks, nthreads, sharedMemSize, stream>>>(x_new, d_sum);
      }
      else
      {
        finalError<<<nblocks, nthreads, sharedMemSize, stream>>>(x, d_sum);
      }

      cuda_safe_call(cudaMemcpyAsync(&sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost, stream));
      cuda_safe_call(cudaStreamSynchronize(stream));
      // printf("GPU iterations : %d\n", k + 1);
      // printf("GPU error: %.3e\n", sum);
      break;
    }
  }

  cuda_safe_call(cudaFree(d_sum));
  return sum;
}

// Run the Jacobi method for A*x = b on GPU without CUDA Graph.
double JacobiMethodGpu(
  const float* A,
  const double* b,
  const float conv_threshold,
  const int max_iter,
  double* x,
  double* x_new,
  cudaStream_t stream)
{
  // CTA size
  dim3 nthreads(256, 1, 1);
  // grid size
  dim3 nblocks((N_ROWS / ROWS_PER_CTA) + 2, 1, 1);

  double sum = 0.0;
  double* d_sum;
  cuda_safe_call(cudaMalloc(&d_sum, sizeof(double)));
  int k = 0;

  for (k = 0; k < max_iter; k++)
  {
    cuda_safe_call(cudaMemsetAsync(d_sum, 0, sizeof(double), stream));
    if ((k & 1) == 0)
    {
      JacobiMethod<<<nblocks, nthreads, 0, stream>>>(A, b, conv_threshold, x, x_new, d_sum);
    }
    else
    {
      JacobiMethod<<<nblocks, nthreads, 0, stream>>>(A, b, conv_threshold, x_new, x, d_sum);
    }
    cuda_safe_call(cudaMemcpyAsync(&sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost, stream));
    cuda_safe_call(cudaStreamSynchronize(stream));

    if (sum <= conv_threshold)
    {
      cuda_safe_call(cudaMemsetAsync(d_sum, 0, sizeof(double), stream));
      nblocks.x            = (N_ROWS / nthreads.x) + 1;
      size_t sharedMemSize = ((nthreads.x / 32) + 1) * sizeof(double);
      if ((k & 1) == 0)
      {
        finalError<<<nblocks, nthreads, sharedMemSize, stream>>>(x_new, d_sum);
      }
      else
      {
        finalError<<<nblocks, nthreads, sharedMemSize, stream>>>(x, d_sum);
      }

      cuda_safe_call(cudaMemcpyAsync(&sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost, stream));
      cuda_safe_call(cudaStreamSynchronize(stream));
      // printf("GPU iterations : %d\n", k + 1);
      // printf("GPU error: %.3e\n", sum);
      break;
    }
  }

  cuda_safe_call(cudaFree(d_sum));
  return sum;
}

// Run the Jacobi method for A*x = b on CPU.
void JacobiMethodCPU(float* A, double* b, float conv_threshold, int max_iter, int* num_iter, double* x)
{
  double* x_new;
  x_new = (double*) calloc(N_ROWS, sizeof(double));
  int k;

  for (k = 0; k < max_iter; k++)
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
  free(x_new);
}

int main()
{
  //  if (checkCmdLineFlag(argc, (const char **)argv, "help")) {
  //    printf("Command line: jacobiCudaGraphs [-option]\n");
  //    printf("Valid options:\n");
  //    printf(
  //        "-gpumethod=<0,1 or 2>  : 0 - [Default] "
  //        "JacobiMethodGpuCudaGraphExecKernelSetParams\n");
  //    printf("                       : 1 - JacobiMethodGpuCudaGraphExecUpdate\n");
  //    printf("                       : 2 - JacobiMethodGpu - Non CUDA Graph\n");
  //    printf("-device=device_num     : cuda device id");
  //    printf("-help         : Output a help message\n");
  //    exit(EXIT_SUCCESS);
  //  }
  //
  int gpumethod = 0;
  //  if (checkCmdLineFlag(argc, (const char **)argv, "gpumethod")) {
  //    gpumethod = getCmdLineArgumentInt(argc, (const char **)argv, "gpumethod");
  //
  //    if (gpumethod < 0 || gpumethod > 2) {
  //      printf("Error: gpumethod must be 0 or 1 or 2, gpumethod=%d is invalid\n",
  //             gpumethod);
  //      exit(EXIT_SUCCESS);
  //    }
  //  }

  //  int dev = findCudaDevice(argc, (const char **)argv);
  // int dev = 0;

  double* b = NULL;
  float* A  = NULL;
  cuda_safe_call(cudaMallocHost(&b, N_ROWS * sizeof(double)));
  memset(b, 0, N_ROWS * sizeof(double));
  cuda_safe_call(cudaMallocHost(&A, N_ROWS * N_ROWS * sizeof(float)));
  memset(A, 0, N_ROWS * N_ROWS * sizeof(float));

  createLinearSystem(A, b);
  double* x = NULL;
  // start with array of all zeroes
  x = (double*) calloc(N_ROWS, sizeof(double));

  float conv_threshold = 1.0e-2;
  int max_iter         = 4 * N_ROWS * N_ROWS;
  int cnt              = 0;

  //  // create timer
  //  StopWatchInterface *timerCPU = NULL, *timerGpu = NULL;
  //  sdkCreateTimer(&timerCPU);
  //
  //  sdkStartTimer(&timerCPU);
  JacobiMethodCPU(A, b, conv_threshold, max_iter, &cnt, x);

  double sum = 0.0;
  // Compute error
  for (int i = 0; i < N_ROWS; i++)
  {
    double d = x[i] - 1.0;
    sum += fabs(d);
  }
  //  sdkStopTimer(&timerCPU);
  // printf("CPU iterations : %d\n", cnt);
  // printf("CPU error: %.3e\n", sum);
  //  printf("CPU Processing time: %f (ms)\n", sdkGetTimerValue(&timerCPU));

  float* d_A;
  double *d_b, *d_x, *d_x_new;
  cudaStream_t stream1;
  cuda_safe_call(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking));
  cuda_safe_call(cudaMalloc(&d_b, sizeof(double) * N_ROWS));
  cuda_safe_call(cudaMalloc(&d_A, sizeof(float) * N_ROWS * N_ROWS));
  cuda_safe_call(cudaMalloc(&d_x, sizeof(double) * N_ROWS));
  cuda_safe_call(cudaMalloc(&d_x_new, sizeof(double) * N_ROWS));

  cuda_safe_call(cudaMemsetAsync(d_x, 0, sizeof(double) * N_ROWS, stream1));
  cuda_safe_call(cudaMemsetAsync(d_x_new, 0, sizeof(double) * N_ROWS, stream1));
  cuda_safe_call(cudaMemcpyAsync(d_A, A, sizeof(float) * N_ROWS * N_ROWS, cudaMemcpyHostToDevice, stream1));
  cuda_safe_call(cudaMemcpyAsync(d_b, b, sizeof(double) * N_ROWS, cudaMemcpyHostToDevice, stream1));

  //  sdkCreateTimer(&timerGpu);
  //  sdkStartTimer(&timerGpu);

  double sumGPU = 0.0;
  if (gpumethod == 0)
  {
    sumGPU = JacobiMethodGpuCudaGraphExecKernelSetParams(d_A, d_b, conv_threshold, max_iter, d_x, d_x_new, stream1);
  }
  else if (gpumethod == 1)
  {
    sumGPU = JacobiMethodGpuCudaGraphExecUpdate(d_A, d_b, conv_threshold, max_iter, d_x, d_x_new, stream1);
  }
  else if (gpumethod == 2)
  {
    sumGPU = JacobiMethodGpu(d_A, d_b, conv_threshold, max_iter, d_x, d_x_new, stream1);
  }

  //  sdkStopTimer(&timerGpu);
  //  printf("GPU Processing time: %f (ms)\n", sdkGetTimerValue(&timerGpu));

  cuda_safe_call(cudaFree(d_b));
  cuda_safe_call(cudaFree(d_A));
  cuda_safe_call(cudaFree(d_x));
  cuda_safe_call(cudaFree(d_x_new));

  cuda_safe_call(cudaFreeHost(A));
  cuda_safe_call(cudaFreeHost(b));

  // printf("&&&& jacobiCudaGraphs %s\n", (fabs(sum - sumGPU) < conv_threshold) ? "PASSED" : "FAILED");

  return (fabs(sum - sumGPU) < conv_threshold) ? EXIT_SUCCESS : EXIT_FAILURE;
}
