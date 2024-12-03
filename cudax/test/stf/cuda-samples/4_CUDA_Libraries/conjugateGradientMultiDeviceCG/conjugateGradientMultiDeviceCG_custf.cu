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

/*
 * This sample implements a conjugate gradient solver on multiple GPU using
 * Unified Memory optimized prefetching and usage hints.
 *
 */

// includes, system
#include <cuda_runtime.h>

#include <iostream>
#include <map>
#include <set>
#include <utility>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Utilities and system includes
#include <cuda/experimental/__stf/places/blocked_partition.cuh>
#include <cuda/experimental/stf.cuh>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

using namespace cuda::experimental::stf;

namespace cg = cooperative_groups;

const char* sSDKname = "conjugateGradientMultiDeviceCG";

#define ENABLE_CPU_DEBUG_CODE 0
#define THREADS_PER_BLOCK     64

__device__ double grid_dot_result = 0.0;

/* genTridiag: generate a random tridiagonal symmetric matrix */
void genTridiag(slice<int> I, slice<int> J, slice<float> val, int N, int nz)
{
  I(0) = 0, J(0) = 0, J(1) = 1;
  val(0) = (float) rand() / RAND_MAX + 10.0f;
  val(1) = (float) rand() / RAND_MAX;
  int start;

  for (int i = 1; i < N; i++)
  {
    if (i > 1)
    {
      I(i) = I(i - 1) + 3;
    }
    else
    {
      I(1) = 2;
    }

    start        = (i - 1) * 3 + 2;
    J(start)     = i - 1;
    J(start + 1) = i;

    if (i < N - 1)
    {
      J(start + 2) = i + 1;
    }

    val(start)     = val(start - 1);
    val(start + 1) = (float) rand() / RAND_MAX + 10.0f;

    if (i < N - 1)
    {
      val(start + 2) = (float) rand() / RAND_MAX;
    }
  }

  I(N) = nz;
}

// I - contains location of the given non-zero element in the row of the matrix
// J - contains location of the given non-zero element in the column of the
// matrix val - contains values of the given non-zero elements of the matrix
// inputVecX - input vector to be multiplied
// outputVecY - resultant vector
void cpuSpMV(int* I, int* J, float* val, int /*unused*/, int num_rows, float alpha, float* inputVecX, float* outputVecY)
{
  for (int i = 0; i < num_rows; i++)
  {
    int num_elems_this_row = I[i + 1] - I[i];

    float output = 0.0;
    for (int j = 0; j < num_elems_this_row; j++)
    {
      output += alpha * val[I[i] + j] * inputVecX[J[I[i] + j]];
    }
    outputVecY[i] = output;
  }

  return;
}

float dotProduct(float* vecA, float* vecB, int size)
{
  float result = 0.0;

  for (int i = 0; i < size; i++)
  {
    result = result + (vecA[i] * vecB[i]);
  }

  return result;
}

void scaleVector(float* vec, float alpha, int size)
{
  for (int i = 0; i < size; i++)
  {
    vec[i] = alpha * vec[i];
  }
}

void saxpy(float* x, float* y, float a, int size)
{
  for (int i = 0; i < size; i++)
  {
    y[i] = a * x[i] + y[i];
  }
}

void cpuConjugateGrad(int* I, int* J, float* val, float* x, float* Ax, float* p, float* r, int nnz, int N, float tol)
{
  int max_iter = 10000;

  float alpha   = 1.0;
  float alpham1 = -1.0;
  float r0      = 0.0, b, a, na;

  cpuSpMV(I, J, val, nnz, N, alpha, x, Ax);
  saxpy(Ax, r, alpham1, N);

  float r1 = dotProduct(r, r, N);

  int k = 1;

  while (r1 > tol * tol && k <= max_iter)
  {
    if (k > 1)
    {
      b = r1 / r0;
      scaleVector(p, b, N);

      saxpy(r, p, alpha, N);
    }
    else
    {
      for (int i = 0; i < N; i++)
      {
        p[i] = r[i];
      }
    }

    cpuSpMV(I, J, val, nnz, N, alpha, p, Ax);

    float dot = dotProduct(p, Ax, N);
    a         = r1 / dot;

    saxpy(p, x, a, N);
    na = -a;
    saxpy(Ax, r, na, N);

    r0 = r1;
    r1 = dotProduct(r, r, N);

    printf("\nCPU code iteration = %3d, residual = %e\n", k, sqrt(r1));
    k++;
  }
}

template <typename thread_hierarchy_t>
__device__ void gpuSpMV(
  slice<int> I,
  slice<int> J,
  slice<float> val,
  int nnz,
  int num_rows,
  float alpha,
  slice<float> inputVecX,
  slice<float> outputVecY,
  const thread_hierarchy_t& t)
{
  for (int i = t.rank(); i < num_rows; i += t.size())
  {
    int row_elem           = I(i);
    int next_row_elem      = I(i + 1);
    int num_elems_this_row = next_row_elem - row_elem;

    float output = 0.0;
    for (int j = 0; j < num_elems_this_row; j++)
    {
      output += alpha * val(row_elem + j) * inputVecX(J(row_elem + j));
    }

    outputVecY(i) = output;
  }
}

template <typename thread_hierarchy_t>
__device__ void gpuSaxpy(slice<float> x, slice<float> y, float a, int size, const thread_hierarchy_t& t)
{
  for (int i = t.rank(); i < size; i += t.size())
  {
    y(i) = a * x(i) + y(i);
  }
}

template <typename thread_hierarchy_t>
__device__ double
gpuDotProduct(slice<float> vecA, slice<float> vecB, int size, double* dot_result, thread_hierarchy_t& t)
{
  slice<double> tmp = t.template storage<double>(1);

  cg::thread_block cta = cooperative_groups::this_thread_block();

  double temp_sum = 0.0;

  for (int i = t.rank(); i < size; i += t.size())
  {
    temp_sum += (double) (vecA(i) * vecB(i));
  }
  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
  temp_sum                         = cg::reduce(tile32, temp_sum, cg::plus<double>());

  if (tile32.thread_rank() == 0)
  {
    tmp[tile32.meta_group_rank()] = temp_sum;
  }

  cta.sync();

  if (tile32.meta_group_rank() == 0)
  {
    temp_sum = tile32.thread_rank() < tile32.meta_group_size() ? tmp[tile32.thread_rank()] : 0.0;
    temp_sum = cg::reduce(tile32, temp_sum, cg::plus<double>());

    if (tile32.thread_rank() == 0)
    {
      atomicAdd(&grid_dot_result, temp_sum);
    }
  }

  t.sync();

  if (t.rank(0, -1) == 0)
  {
    atomicAdd_system(dot_result, grid_dot_result);
    grid_dot_result = 0.0;
  }

  t.sync();
  return *dot_result;
}

template <typename thread_hierarchy_t>
__device__ void gpuCopyVector(slice<float> srcA, slice<float> destB, int size, const thread_hierarchy_t& t)
{
  for (int i = t.rank(); i < size; i += t.size())
  {
    destB(i) = srcA(i);
  }
}

template <typename thread_hierarchy_t>
__device__ void
gpuScaleVectorAndSaxpy(slice<float> x, slice<float> y, float a, float scale, int size, const thread_hierarchy_t& t)
{
  for (int i = t.rank(); i < size; i += t.size())
  {
    y(i) = a * x(i) + scale * y(i);
  }
}

template <typename thread_hierarchy_t>
__device__ void multiGpuConjugateGradient(
  thread_hierarchy_t t,
  slice<int> I,
  slice<int> J,
  slice<float> val,
  slice<float> x,
  slice<float> Ax,
  slice<float> p,
  slice<float> r,
  double* dot_result,
  int nnz,
  int N,
  float tol)
{
  const int max_iter = 10000;

  float alpha   = 1.0;
  float alpham1 = -1.0;
  float r0      = 0.0, r1, b, a, na;

  for (int i = t.rank(); i < N; i += t.size())
  {
    r[i] = 1.0;
    x[i] = 0.0;
  }

  gpuSpMV(I, J, val, nnz, N, alpha, x, Ax, t);

  gpuSaxpy(Ax, r, alpham1, N, t);

  r1 = gpuDotProduct(r, r, N, dot_result, t);

  int k = 1;
  while (r1 > tol * tol && k <= max_iter)
  {
    if (k > 1)
    {
      b = r1 / r0;
      gpuScaleVectorAndSaxpy(r, p, alpha, b, N, t);
    }
    else
    {
      gpuCopyVector(r, p, N, t);
    }

    gpuSpMV(I, J, val, nnz, N, alpha, p, Ax, t);

    if (t.rank() == 0)
    {
      *dot_result = 0.0;
    }

    a = r1 / gpuDotProduct(p, Ax, N, dot_result, t);

    gpuSaxpy(p, x, a, N, t);

    na = -a;

    gpuSaxpy(Ax, r, na, N, t);

    r0 = r1;

    if (t.rank() == 0)
    {
      *dot_result = 0.0;
    }

    r1 = gpuDotProduct(r, r, N, dot_result, t);

    k++;
  }
}

int main()
{
  stream_ctx ctx;
#if 0
    constexpr size_t kNumGpusRequired = 8;
#else
  constexpr size_t kNumGpusRequired = 1;
#endif
  int N = 0, nz = 0, *I = NULL, *J = NULL;
  float* val      = NULL;
  const float tol = 1e-5f;
  float* x;
  float rhs = 1.0;
  float r1;
  float *r, *p, *Ax;

  // printf("Starting [%s]...\n", sSDKname);

  /* Generate a random tridiagonal symmetric matrix in CSR format */
  N  = 10485760 * 2;
  nz = (N - 2) * 3 + 4;

  I              = (int*) malloc(sizeof(int) * (N + 1));
  J              = (int*) malloc(sizeof(int) * nz);
  val            = (float*) malloc(sizeof(float) * nz);
  float* val_cpu = (float*) malloc(sizeof(float) * nz);

  auto handle_I   = ctx.logical_data(I, {(unsigned) (N + 1)});
  auto handle_J   = ctx.logical_data(J, {(unsigned) nz});
  auto handle_val = ctx.logical_data(val, {(unsigned) nz});

  ctx.host_launch(handle_I.write(), handle_J.write(), handle_val.write())->*[=](auto I, auto J, auto val) {
    genTridiag(I, J, val, N, nz);
    memcpy(val_cpu, val.data_handle(), sizeof(float) * nz);
  };

  double* dot_result = (double*) malloc(sizeof(double));
  dot_result[0]      = 0.0;

  x  = (float*) malloc(sizeof(float) * N);
  r  = (float*) malloc(sizeof(float) * N);
  p  = (float*) malloc(sizeof(float) * N);
  Ax = (float*) malloc(sizeof(float) * N);

  auto handle_r          = ctx.logical_data(r, {(unsigned) N});
  auto handle_p          = ctx.logical_data(p, {(unsigned) N});
  auto handle_Ax         = ctx.logical_data(Ax, {(unsigned) N});
  auto handle_x          = ctx.logical_data(x, {(unsigned) N});
  auto handle_dot_result = ctx.logical_data(dot_result, {(unsigned) 1});

  // std::cout << "\nRunning on GPUs = " << kNumGpusRequired << std::endl;
  const int sMemSize = sizeof(double) * ((THREADS_PER_BLOCK / 32) + 1);

  // auto all_devs = exec_place::repeat<blocked_partition>(exec_place::device(0), kNumGpusRequired);
  auto all_devs = exec_place::n_devices(kNumGpusRequired);

  /* The grid size is 0 and will be computed upon launch */
  auto spec = con(con(THREADS_PER_BLOCK, mem(sMemSize)));
  ctx.launch(
    spec,
    all_devs,
    handle_I.read(),
    handle_J.read(),
    handle_val.read(),
    handle_x.write(),
    handle_Ax.write(),
    handle_p.write(),
    handle_r.write(),
    handle_dot_result.write())
      ->*[=]
    _CCCL_DEVICE(auto t,
                 slice<int> I,
                 slice<int> J,
                 slice<float> val,
                 slice<float> x,
                 slice<float> Ax,
                 slice<float> p,
                 slice<float> r,
                 slice<double> dot_result) {
      multiGpuConjugateGradient(t, I, J, val, x, Ax, p, r, dot_result.data_handle(), nz, N, tol);
    };

  ctx.finalize();

  r1 = dot_result[0];

  printf("GPU Final, residual = %e \n  ", sqrt(r1));

#if ENABLE_CPU_DEBUG_CODE
  float* Ax_cpu = (float*) malloc(sizeof(float) * N);
  float* r_cpu  = (float*) malloc(sizeof(float) * N);
  float* p_cpu  = (float*) malloc(sizeof(float) * N);
  float* x_cpu  = (float*) malloc(sizeof(float) * N);

  for (int i = 0; i < N; i++)
  {
    r_cpu[i]  = 1.0;
    Ax_cpu[i] = x_cpu[i] = 0.0;
  }
  cpuConjugateGrad(I, J, val, x_cpu, Ax_cpu, p_cpu, r_cpu, nz, N, tol);
#endif

  float rsum, diff, err = 0.0;

  for (int i = 0; i < N; i++)
  {
    rsum = 0.0;

    for (int j = I[i]; j < I[i + 1]; j++)
    {
      rsum += val_cpu[j] * x[J[j]];
    }

    diff = fabs(rsum - rhs);

    if (diff > err)
    {
      err = diff;
    }
  }

#if ENABLE_CPU_DEBUG_CODE
  free(Ax_cpu);
  free(r_cpu);
  free(p_cpu);
  free(x_cpu);
#endif

  printf("Test Summary:  Error amount = %f \n", err);
  fprintf(stdout, "&&&& conjugateGradientMultiDeviceCG %s\n", (sqrt(r1) < tol) ? "PASSED" : "FAILED");
  exit((sqrt(r1) < tol) ? EXIT_SUCCESS : EXIT_FAILURE);
}
