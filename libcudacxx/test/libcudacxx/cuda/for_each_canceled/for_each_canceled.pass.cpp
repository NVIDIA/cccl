//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: pre-sm-100
// UNSUPPORTED: !nvcc
// UNSUPPORTED: nvrtc
// UNSUPPORTED: c++98, c++03, c++14, c++17

#include <cuda/for_each_canceled>

__device__ void vec_add_impl1(int* a, int* b, int* c, int n, dim3 block_idx)
{
  int idx = threadIdx.x + block_idx.x * blockDim.x;
  if (idx < n)
  {
    c[idx] += a[idx] + b[idx];
  }
}

__device__ void vec_add_impl2(int* a, int* b, int* c, int n, dim3 block_idx)
{
  int x_idx = threadIdx.x + (block_idx.x * blockDim.x);
  int y_idx = threadIdx.y + (block_idx.y * blockDim.y);
  int idx   = x_idx + y_idx * (blockDim.x * gridDim.x);
  if (idx < n)
  {
    c[idx] += a[idx] + b[idx];
  }
}

__device__ void vec_add_impl3(int* a, int* b, int* c, int n, dim3 block_idx)
{
  int x_idx = threadIdx.x + (block_idx.x * blockDim.x);
  int y_idx = threadIdx.y + (block_idx.y * blockDim.y);
  int z_idx = threadIdx.z + (block_idx.z * blockDim.z);
  int idx   = x_idx + y_idx * (blockDim.x * gridDim.x) + z_idx * (blockDim.x * gridDim.x * blockDim.y * gridDim.y);
  if (idx < n)
  {
    c[idx] += a[idx] + b[idx];
  }
}

__global__ void vec_add_det1(int* a, int* b, int* c, int n, int tidx = 0)
{
  cuda::__detail::__for_each_canceled_block<1>(threadIdx.x == tidx, [=](dim3 block_idx) {
    vec_add_impl1(a, b, c, n, block_idx);
  });
}

__global__ void vec_add_det2(int* a, int* b, int* c, int n, int tidx = 0)
{
  cuda::__detail::__for_each_canceled_block<2>(
    threadIdx.x == tidx && threadIdx.y == tidx, [=](dim3 block_idx) {
      vec_add_impl2(a, b, c, n, block_idx);
    });
}

__global__ void vec_add_det3(int* a, int* b, int* c, int n, int tidx = 0)
{
  cuda::__detail::__for_each_canceled_block<3>(
    threadIdx.x == tidx && threadIdx.y == tidx && threadIdx.z == tidx, [=](dim3 block_idx) {
      vec_add_impl3(a, b, c, n, block_idx);
    });
}

__global__ void vec_add1(int* a, int* b, int* c, int n, int tidx = 0)
{
  cuda::for_each_canceled_block<1>([=](dim3 block_idx) {
    vec_add_impl1(a, b, c, n, block_idx);
  });
}

__global__ void vec_add2(int* a, int* b, int* c, int n, int tidx = 0)
{
  cuda::for_each_canceled_block<2>([=](dim3 block_idx) {
    vec_add_impl2(a, b, c, n, block_idx);
  });
}

__global__ void vec_add3(int* a, int* b, int* c, int n, int tidx = 0)
{
  cuda::for_each_canceled_block<3>([=](dim3 block_idx) {
    vec_add_impl3(a, b, c, n, block_idx);
  });
}

#include <iostream>
#include <vector>

template <typename F>
bool test(size_t N, F&& f)
{
  for (int tidx : {0, 33, 63, 94})
  {
    int *a, *b, *c;
    cudaMallocManaged(&a, N * sizeof(int));
    cudaMallocManaged(&b, N * sizeof(int));
    cudaMallocManaged(&c, N * sizeof(int));
    for (int i = 0; i < N; ++i)
    {
      a[i] = i;
      b[i] = 1;
      c[i] = 0;
    }

    f(a, b, c, N, tidx);
    if (auto e = cudaDeviceSynchronize(); e != cudaSuccess)
    {
      std::cerr << "ERROR: synchronize failed" << std::endl;
      return false;
    }

    bool success = true;
    for (int i = 0; i < N; ++i)
    {
      if (c[i] != (1 + i))
      {
        std::cerr << "ERROR " << i << ", " << c[i] << std::endl;
        success = false;
      }
    }
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    if (!success)
    {
      return false;
    }
  }
  return true;
}

int main(int argc, char** argv)
{
  NV_IF_TARGET(
    NV_IS_HOST,
    (size_t N = 1000000;
     if (!test(N,
               [](int* a, int* b, int* c, size_t n, int tidx) {
                 int tpb = 256;
                 int bpg = (n + tpb - 1) / tpb;
                 vec_add_det1<<<bpg, tpb>>>(a, b, c, n, tidx);
               })) return 1;

     if (!test(N,
               [](int* a, int* b, int* c, size_t n, int tidx) {
                 dim3 tpb(16, 16, 1);
                 int ntpb = tpb.x * tpb.y; // 256
                 int bpg  = (n + ntpb - 1) / ntpb;
                 int bpgx = (int) std::sqrt((double) bpg) + 1;
                 int bpgy = bpgx;
                 if ((bpgx * bpgy) < bpg)
                 {
                   std::cerr << "ERROR in grid" << std::endl;
                   std::abort();
                 }
                 vec_add_det2<<<dim3(bpgx, bpgy, 1), tpb>>>(a, b, c, n, tidx);
               })) return 1;

     if (!test(N,
               [](int* a, int* b, int* c, size_t n, int tidx) {
                 dim3 tpb(8, 8, 8);
                 int ntpb = tpb.x * tpb.y * tpb.z; // 512
                 int bpg  = (n + ntpb - 1) / ntpb;
                 int bpgx = (int) std::pow((double) bpg, 1. / 3.) + 1;
                 int bpgy = bpgx;
                 int bpgz = bpgx;
                 if ((bpgx * bpgy * bpgz) < bpg)
                 {
                   std::cerr << "ERROR in grid" << std::endl;
                   std::abort();
                 }
                 vec_add_det3<<<dim3(bpgx, bpgy, bpgz), tpb>>>(a, b, c, n, tidx);
               })) return 1;

     if (!test(N,
               [](int* a, int* b, int* c, size_t n, int tidx) {
                 int tpb = 256;
                 int bpg = (n + tpb - 1) / tpb;
                 vec_add1<<<bpg, tpb>>>(a, b, c, n);
               })) return 1;

     if (!test(N,
               [](int* a, int* b, int* c, size_t n, int tidx) {
                 dim3 tpb(16, 16, 1);
                 int ntpb = tpb.x * tpb.y; // 256
                 int bpg  = (n + ntpb - 1) / ntpb;
                 int bpgx = (int) std::sqrt((double) bpg) + 1;
                 int bpgy = bpgx;
                 if ((bpgx * bpgy) < bpg)
                 {
                   std::cerr << "ERROR in grid" << std::endl;
                   std::abort();
                 }
                 vec_add2<<<dim3(bpgx, bpgy, 1), tpb>>>(a, b, c, n);
               })) return 1;

     if (!test(N, [](int* a, int* b, int* c, size_t n, int tidx) {
           dim3 tpb(8, 8, 8);
           int ntpb = tpb.x * tpb.y * tpb.z; // 512
           int bpg  = (n + ntpb - 1) / ntpb;
           int bpgx = (int) std::pow((double) bpg, 1. / 3.) + 1;
           int bpgy = bpgx;
           int bpgz = bpgx;
           if ((bpgx * bpgy * bpgz) < bpg)
           {
             std::cerr << "ERROR in grid" << std::endl;
             std::abort();
           }
           vec_add3<<<dim3(bpgx, bpgy, bpgz), tpb>>>(a, b, c, n);
         })) return 1;));

  return 0;
}
