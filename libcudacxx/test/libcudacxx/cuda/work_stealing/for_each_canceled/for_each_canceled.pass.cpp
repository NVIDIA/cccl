//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: !nvcc
// UNSUPPORTED: nvrtc

#include <cuda/std/cmath>
#include <cuda/work_stealing>

#if _CCCL_HAS_INT128()

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

__global__ void vec_add_det1(int* a, int* b, int* c, int n, int leader_tidx = 0)
{
  ::cuda::__for_each_canceled_block<1>(threadIdx.x == leader_tidx, [=](dim3 block_idx) {
    vec_add_impl1(a, b, c, n, block_idx);
  });
}

__global__ void vec_add_det2(int* a, int* b, int* c, int n, int leader_tidx = 0)
{
  ::cuda::__for_each_canceled_block<2>(threadIdx.x == leader_tidx && threadIdx.y == leader_tidx, [=](dim3 block_idx) {
    vec_add_impl2(a, b, c, n, block_idx);
  });
}

__global__ void vec_add_det3(int* a, int* b, int* c, int n, int leader_tidx = 0)
{
  ::cuda::__for_each_canceled_block<3>(
    threadIdx.x == leader_tidx && threadIdx.y == leader_tidx && threadIdx.z == leader_tidx, [=](dim3 block_idx) {
      vec_add_impl3(a, b, c, n, block_idx);
    });
}

__global__ void vec_add1(int* a, int* b, int* c, int n)
{
  cuda::for_each_canceled_block<1>([=](dim3 block_idx) {
    vec_add_impl1(a, b, c, n, block_idx);
  });
}

__global__ void vec_add2(int* a, int* b, int* c, int n)
{
  cuda::for_each_canceled_block<2>([=](dim3 block_idx) {
    vec_add_impl2(a, b, c, n, block_idx);
  });
}

__global__ void vec_add3(int* a, int* b, int* c, int n)
{
  cuda::for_each_canceled_block<3>([=](dim3 block_idx) {
    vec_add_impl3(a, b, c, n, block_idx);
  });
}

template <typename F>
bool test(int N, F&& f)
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
    assert(cudaDeviceSynchronize() == cudaSuccess);

    for (int i = 0; i < N; ++i)
    {
      assert(c[i] == (1 + i));
    }

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
  }
  return true;
}

void test()
{
  const int N = 1000000;
  {
    auto fn = [](int* a, int* b, int* c, int n, int tidx) {
      int tpb = 256;
      int bpg = (n + tpb - 1) / tpb;
      vec_add_det1<<<bpg, tpb>>>(a, b, c, n, tidx);
    };
    assert(test(N, fn));
  }

  {
    auto fn = [](int* a, int* b, int* c, int n, int tidx) {
      dim3 tpb(16, 16, 1);
      int ntpb = tpb.x * tpb.y; // 256
      int bpg  = (n + ntpb - 1) / ntpb;
      int bpgx = (int) cuda::std::sqrt(bpg) + 1;
      int bpgy = bpgx;
      assert((bpgx * bpgy) >= bpg);
      vec_add_det2<<<dim3(bpgx, bpgy, 1), tpb>>>(a, b, c, n, tidx);
    };
    assert(test(N, fn));
  }

  {
    auto fn = [](int* a, int* b, int* c, int n, int tidx) {
      dim3 tpb(8, 8, 8);
      int ntpb = tpb.x * tpb.y * tpb.z; // 512
      int bpg  = (n + ntpb - 1) / ntpb;
      int bpgx = (int) cuda::std::pow(bpg, 1. / 3.) + 1;
      int bpgy = bpgx;
      int bpgz = bpgx;
      assert((bpgx * bpgy * bpgz) >= bpg);
      vec_add_det3<<<dim3(bpgx, bpgy, bpgz), tpb>>>(a, b, c, n, tidx);
    };
    assert(test(N, fn));
  }

  {
    auto fn = [](int* a, int* b, int* c, int n, int tidx) {
      int tpb = 256;
      int bpg = (n + tpb - 1) / tpb;
      vec_add1<<<bpg, tpb>>>(a, b, c, n);
    };
    assert(test(N, fn));
  }

  {
    auto fn = [](int* a, int* b, int* c, int n, int tidx) {
      dim3 tpb(16, 16, 1);
      int ntpb = tpb.x * tpb.y; // 256
      int bpg  = (n + ntpb - 1) / ntpb;
      int bpgx = (int) cuda::std::sqrt(bpg) + 1;
      int bpgy = bpgx;
      assert((bpgx * bpgy) >= bpg);
      vec_add2<<<dim3(bpgx, bpgy, 1), tpb>>>(a, b, c, n);
    };
    assert(test(N, fn));
  }

  {
    auto fn = [](int* a, int* b, int* c, int n, int tidx) {
      dim3 tpb(8, 8, 8);
      int ntpb = tpb.x * tpb.y * tpb.z; // 512
      int bpg  = (n + ntpb - 1) / ntpb;
      int bpgx = (int) cuda::std::pow(bpg, 1. / 3.) + 1;
      int bpgy = bpgx;
      int bpgz = bpgx;
      assert((bpgx * bpgy * bpgz) >= bpg);
      vec_add3<<<dim3(bpgx, bpgy, bpgz), tpb>>>(a, b, c, n);
    };
    assert(test(N, fn));
  }
}

#endif // _CCCL_HAS_INT128()

int main(int argc, char** argv)
{
#if _CCCL_HAS_INT128()
  NV_IF_TARGET(NV_IS_HOST, (test();))
#endif // _CCCL_HAS_INT128()

  return 0;
}
