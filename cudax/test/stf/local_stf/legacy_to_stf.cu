//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/__stf/utility/nvtx.cuh>
#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

__global__ void initA(double* d_ptrA, size_t N)
{
  size_t tid      = blockIdx.x * blockDim.x + threadIdx.x;
  size_t nthreads = blockDim.x * gridDim.x;
  for (size_t i = tid; i < N; i += nthreads)
  {
    d_ptrA[i] = sin((double) i);
  }
}

__global__ void initB(double* d_ptrB, size_t N)
{
  size_t tid      = blockIdx.x * blockDim.x + threadIdx.x;
  size_t nthreads = blockDim.x * gridDim.x;
  for (size_t i = tid; i < N; i += nthreads)
  {
    d_ptrB[i] = cos((double) i);
  }
}

// B += alpha*A;
__global__ void axpy(double alpha, const double* d_ptrA, double* d_ptrB, size_t N)
{
  int tid      = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;

  for (int i = tid; i < N; i += nthreads)
  {
    d_ptrB[i] += alpha * d_ptrA[i];
  }
}

__global__ void empty_kernel()
{
  // no-op
}

double X0(int i)
{
  return sin((double) i);
}

double Y0(int i)
{
  return cos((double) i);
}

void ref_lib_call(cudaStream_t stream, double* d_ptrA, double* d_ptrB, size_t N)
{
  initA<<<128, 32, 0, stream>>>(d_ptrA, N);
  initB<<<128, 32, 0, stream>>>(d_ptrB, N);
  axpy<<<128, 32, 0, stream>>>(3.0, d_ptrA, d_ptrB, N);
  empty_kernel<<<16, 8, 0, stream>>>();
}

void lib_call(cudaStream_t stream, double* d_ptrA, double* d_ptrB, size_t N)
{
  stream_ctx ctx(stream);
  auto lA = ctx.logical_data(make_slice(d_ptrA, N), data_place::current_device());
  auto lB = ctx.logical_data(make_slice(d_ptrB, N), data_place::current_device());
  ctx.task(lA.write())->*[=](cudaStream_t s, auto a) {
    initA<<<128, 32, 0, s>>>(a.data_handle(), N);
  };

  ctx.task(lB.write())->*[=](cudaStream_t s, auto b) {
    initB<<<128, 32, 0, s>>>(b.data_handle(), N);
  };

  ctx.task(lA.read(), lB.rw())->*[=](cudaStream_t s, auto a, auto b) {
    axpy<<<128, 32, 0, s>>>(3.0, a.data_handle(), b.data_handle(), N);
  };

  ctx.task()->*[](cudaStream_t s) {
    empty_kernel<<<16, 8, 0, s>>>();
  };

  // Note that this is non-blocking because we have creating the stream_ctx
  // relative to a user-provided CUDA stream
  ctx.finalize();
}

void lib_call_with_handle(async_resources_handle& handle, cudaStream_t stream, double* d_ptrA, double* d_ptrB, size_t N)
{
  stream_ctx ctx(stream, handle);
  auto lA = ctx.logical_data(make_slice(d_ptrA, N), data_place::current_device());
  auto lB = ctx.logical_data(make_slice(d_ptrB, N), data_place::current_device());
  ctx.task(lA.write())->*[=](cudaStream_t s, auto a) {
    initA<<<128, 32, 0, s>>>(a.data_handle(), N);
  };

  ctx.task(lB.write())->*[=](cudaStream_t s, auto b) {
    initB<<<128, 32, 0, s>>>(b.data_handle(), N);
  };

  ctx.task(lA.read(), lB.rw())->*[=](cudaStream_t s, auto a, auto b) {
    axpy<<<128, 32, 0, s>>>(3.0, a.data_handle(), b.data_handle(), N);
  };

  ctx.task()->*[](cudaStream_t s) {
    empty_kernel<<<16, 8, 0, s>>>();
  };

  // Note that this is non-blocking because we have creating the stream_ctx
  // relative to a user-provided CUDA stream
  ctx.finalize();
}

template <typename Ctx_t>
void lib_call_generic(async_resources_handle& handle, cudaStream_t stream, double* d_ptrA, double* d_ptrB, size_t N)
{
  Ctx_t ctx(stream, handle);
  auto lA = ctx.logical_data(make_slice(d_ptrA, N), data_place::current_device());
  auto lB = ctx.logical_data(make_slice(d_ptrB, N), data_place::current_device());
  ctx.task(lA.write())->*[=](cudaStream_t s, auto a) {
    initA<<<128, 32, 0, s>>>(a.data_handle(), N);
  };

  ctx.task(lB.write())->*[=](cudaStream_t s, auto b) {
    initB<<<128, 32, 0, s>>>(b.data_handle(), N);
  };

  ctx.task(lA.read(), lB.rw())->*[=](cudaStream_t s, auto a, auto b) {
    axpy<<<128, 32, 0, s>>>(3.0, a.data_handle(), b.data_handle(), N);
  };

  ctx.task()->*[](cudaStream_t s) {
    empty_kernel<<<16, 8, 0, s>>>();
  };

  ctx.submit();
}

template <typename Ctx_t>
void lib_call_token(async_resources_handle& handle, cudaStream_t stream, double* d_ptrA, double* d_ptrB, size_t N)
{
  Ctx_t ctx(stream, handle);
  auto lA = ctx.token();
  auto lB = ctx.token();
  ctx.task(lA.write())->*[=](cudaStream_t s) {
    initA<<<128, 32, 0, s>>>(d_ptrA, N);
  };

  ctx.task(lB.write())->*[=](cudaStream_t s) {
    initB<<<128, 32, 0, s>>>(d_ptrB, N);
  };

  ctx.task(lA.read(), lB.rw())->*[=](cudaStream_t s) {
    axpy<<<128, 32, 0, s>>>(3.0, d_ptrA, d_ptrB, N);
  };

  ctx.task()->*[](cudaStream_t s) {
    empty_kernel<<<16, 8, 0, s>>>();
  };

  ctx.submit();
}

int main()
{
  double *d_ptrA, *d_ptrB;
  const size_t N     = 128 * 1024;
  const size_t NITER = 128;

  // User allocated memory
  cuda_safe_call(cudaMalloc(&d_ptrA, N * sizeof(double)));
  cuda_safe_call(cudaMalloc(&d_ptrB, N * sizeof(double)));

  cudaStream_t stream;
  cuda_safe_call(cudaStreamCreate(&stream));

  nvtx_range r_warmup("warmup");
  for (size_t i = 0; i < NITER; i++)
  {
    ref_lib_call(stream, d_ptrA, d_ptrB, N);
  }
  cuda_safe_call(cudaStreamSynchronize(stream));
  r_warmup.end();

  nvtx_range r_ref("ref");
  for (size_t i = 0; i < NITER; i++)
  {
    ref_lib_call(stream, d_ptrA, d_ptrB, N);
  }
  cuda_safe_call(cudaStreamSynchronize(stream));
  r_ref.end();

  nvtx_range r_local("local stf");
  for (size_t i = 0; i < NITER; i++)
  {
    lib_call(stream, d_ptrA, d_ptrB, N);
  }
  cuda_safe_call(cudaStreamSynchronize(stream));
  r_local.end();

  nvtx_range r_local_handle("local stf handle");
  async_resources_handle handle;
  for (size_t i = 0; i < NITER; i++)
  {
    lib_call_with_handle(handle, stream, d_ptrA, d_ptrB, N);
  }
  cuda_safe_call(cudaStreamSynchronize(stream));
  r_local_handle.end();

  nvtx_range r_generic_graph_handle("generic graph handle");
  for (size_t i = 0; i < NITER; i++)
  {
    lib_call_generic<graph_ctx>(handle, stream, d_ptrA, d_ptrB, N);
  }
  cuda_safe_call(cudaStreamSynchronize(stream));
  r_generic_graph_handle.end();

  nvtx_range r_generic_stream_handle("generic stream handle");
  for (size_t i = 0; i < NITER; i++)
  {
    lib_call_generic<stream_ctx>(handle, stream, d_ptrA, d_ptrB, N);
  }
  cuda_safe_call(cudaStreamSynchronize(stream));
  r_generic_stream_handle.end();

  nvtx_range r_generic_context_handle("generic context handle");
  for (size_t i = 0; i < NITER; i++)
  {
    lib_call_generic<context>(handle, stream, d_ptrA, d_ptrB, N);
  }
  cuda_safe_call(cudaStreamSynchronize(stream));
  r_generic_context_handle.end();

  nvtx_range r_token("logical token");
  for (size_t i = 0; i < NITER; i++)
  {
    lib_call_token<context>(handle, stream, d_ptrA, d_ptrB, N);
  }
  cuda_safe_call(cudaStreamSynchronize(stream));
  r_token.end();
}
