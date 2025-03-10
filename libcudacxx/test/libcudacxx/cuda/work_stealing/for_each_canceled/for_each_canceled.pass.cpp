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

#include <cuda/atomic>
#include <cuda/std/cmath>
#include <cuda/work_stealing>

#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#if !defined(TEST_HAS_NO_INT128_T)

__device__ int vec_add_impl1(int* a, int* b, int* c, int n, dim3 block_idx)
{
  int idx = threadIdx.x + block_idx.x * blockDim.x;
  if (idx < n)
  {
    int s = a[idx] + b[idx];
    c[idx] += s;
    return s;
  }
  return 0;
}

__device__ int vec_add_impl2(int* a, int* b, int* c, int n, dim3 block_idx)
{
  int x_idx = threadIdx.x + (block_idx.x * blockDim.x);
  int y_idx = threadIdx.y + (block_idx.y * blockDim.y);
  int idx   = x_idx + y_idx * (blockDim.x * gridDim.x);
  if (idx < n)
  {
    int s = a[idx] + b[idx];
    c[idx] += s;
    return s;
  }
  return 0;
}

__device__ int vec_add_impl3(int* a, int* b, int* c, int n, dim3 block_idx)
{
  int x_idx = threadIdx.x + (block_idx.x * blockDim.x);
  int y_idx = threadIdx.y + (block_idx.y * blockDim.y);
  int z_idx = threadIdx.z + (block_idx.z * blockDim.z);
  int idx   = x_idx + y_idx * (blockDim.x * gridDim.x) + z_idx * (blockDim.x * gridDim.x * blockDim.y * gridDim.y);
  if (idx < n)
  {
    int s = a[idx] + b[idx];
    c[idx] += s;
    return s;
  }
  return 0;
}

////////////////////////////////////////////////////////////////////////////////
// for_each_cancel_block tests:

__global__ void vec_add_det1(int* s, int* a, int* b, int* c, int n, int leader_tidx = 0)
{
  int thread_sum = 0;
  bool leader    = threadIdx.x == leader_tidx;
  ::cuda::__for_each_canceled_block<1>(leader, [&](dim3 block_idx) {
    thread_sum += vec_add_impl1(a, b, c, n, block_idx);
  });
  auto block = cg::this_thread_block();
  auto tile  = cg::tiled_partition<32>(block);
  cg::reduce_update_async(
    cg::tiled_partition<32>(cg::this_thread_block()), cuda::atomic_ref{*s}, thread_sum, cg::plus<int>{});
}

__global__ void vec_add_det2(int* s, int* a, int* b, int* c, int n, int leader_tidx = 0)
{
  int thread_sum = 0;
  bool leader    = threadIdx.x == leader_tidx && threadIdx.y == leader_tidx;
  ::cuda::__for_each_canceled_block<2>(leader, [&](dim3 block_idx) {
    thread_sum += vec_add_impl2(a, b, c, n, block_idx);
  });
  auto block = cg::this_thread_block();
  auto tile  = cg::tiled_partition<32>(block);
  cg::reduce_update_async(
    cg::tiled_partition<32>(cg::this_thread_block()), cuda::atomic_ref{*s}, thread_sum, cg::plus<int>{});
}

__global__ void vec_add_det3(int* s, int* a, int* b, int* c, int n, int leader_tidx = 0)
{
  int thread_sum = 0;
  bool leader    = threadIdx.x == leader_tidx && threadIdx.y == leader_tidx && threadIdx.z == leader_tidx;
  ::cuda::__for_each_canceled_block<3>(leader, [&](dim3 block_idx) {
    thread_sum += vec_add_impl3(a, b, c, n, block_idx);
  });
  auto block = cg::this_thread_block();
  auto tile  = cg::tiled_partition<32>(block);
  cg::reduce_update_async(
    cg::tiled_partition<32>(cg::this_thread_block()), cuda::atomic_ref{*s}, thread_sum, cg::plus<int>{});
}

__global__ void vec_add1(int* s, int* a, int* b, int* c, int n)
{
  int thread_sum = 0;
  cuda::for_each_canceled_block<1>([&](dim3 block_idx) {
    thread_sum += vec_add_impl1(a, b, c, n, block_idx);
  });
  auto block = cg::this_thread_block();
  auto tile  = cg::tiled_partition<32>(block);
  cg::reduce_update_async(
    cg::tiled_partition<32>(cg::this_thread_block()), cuda::atomic_ref{*s}, thread_sum, cg::plus<int>{});
}

__global__ void vec_add2(int* s, int* a, int* b, int* c, int n)
{
  int thread_sum = 0;
  cuda::for_each_canceled_block<2>([&](dim3 block_idx) {
    thread_sum += vec_add_impl2(a, b, c, n, block_idx);
  });
  auto block = cg::this_thread_block();
  auto tile  = cg::tiled_partition<32>(block);
  cg::reduce_update_async(
    cg::tiled_partition<32>(cg::this_thread_block()), cuda::atomic_ref{*s}, thread_sum, cg::plus<int>{});
}

__global__ void vec_add3(int* s, int* a, int* b, int* c, int n)
{
  int thread_sum = 0;
  cuda::for_each_canceled_block<3>([&](dim3 block_idx) {
    thread_sum += vec_add_impl3(a, b, c, n, block_idx);
  });
  cg::reduce_update_async(
    cg::tiled_partition<32>(cg::this_thread_block()), cuda::atomic_ref{*s}, thread_sum, cg::plus<int>{});
}

////////////////////////////////////////////////////////////////////////////////
// for_each_cancel_cluster tests:

constexpr int CLUSTER_DIM1 = 8;
constexpr int CLUSTER_DIM2 = 2;
constexpr int CLUSTER_DIM3 = 2;

__global__ void __cluster_dims__(CLUSTER_DIM1, 1, 1)
  cluster_vec_add_det1(int* s, int* a, int* b, int* c, int n, int leader_bidx, int leader_tidx)
{
  dim3 local_block_idx = cooperative_groups::cluster_group::block_index();
  int thread_sum       = 0;
  bool leader_block    = local_block_idx.x == leader_bidx;
  bool leader_threads  = threadIdx.x == leader_tidx;
  ::cuda::__for_each_canceled_cluster<1>(leader_block, leader_threads, [&](dim3 block_idx) {
    thread_sum += vec_add_impl1(a, b, c, n, block_idx);
  });
  cg::reduce_update_async(
    cg::tiled_partition<32>(cg::this_thread_block()), cuda::atomic_ref{*s}, thread_sum, cg::plus<int>{});
}

__global__ void __cluster_dims__(CLUSTER_DIM2, CLUSTER_DIM2, 1)
  cluster_vec_add_det2(int* s, int* a, int* b, int* c, int n, int leader_bidx, int leader_tidx)
{
  dim3 local_block_idx = cooperative_groups::cluster_group::block_index();
  int thread_sum       = 0;
  bool leader_block    = local_block_idx.x == leader_bidx && local_block_idx.y == leader_bidx;
  bool leader_threads  = threadIdx.x == leader_tidx && threadIdx.y == leader_tidx;
  ::cuda::__for_each_canceled_cluster<2>(leader_block, leader_threads, [&](dim3 block_idx) {
    thread_sum += vec_add_impl2(a, b, c, n, block_idx);
  });
  cg::reduce_update_async(
    cg::tiled_partition<32>(cg::this_thread_block()), cuda::atomic_ref{*s}, thread_sum, cg::plus<int>{});
}

__global__ void __cluster_dims__(CLUSTER_DIM3, CLUSTER_DIM3, CLUSTER_DIM3)
  cluster_vec_add_det3(int* s, int* a, int* b, int* c, int n, int leader_bidx, int leader_tidx)
{
  dim3 local_block_idx = cooperative_groups::cluster_group::block_index();
  int thread_sum       = 0;
  bool leader_block =
    local_block_idx.x == leader_bidx && local_block_idx.y == leader_bidx && local_block_idx.z == leader_bidx;
  bool leader_threads = threadIdx.x == leader_tidx && threadIdx.y == leader_tidx && threadIdx.z == leader_tidx;
  ::cuda::__for_each_canceled_cluster<3>(leader_block, leader_threads, [&](dim3 block_idx) {
    thread_sum += vec_add_impl3(a, b, c, n, block_idx);
  });
  cg::reduce_update_async(
    cg::tiled_partition<32>(cg::this_thread_block()), cuda::atomic_ref{*s}, thread_sum, cg::plus<int>{});
}

__global__ void __cluster_dims__(CLUSTER_DIM1, 1, 1) cluster_vec_add1(int* s, int* a, int* b, int* c, int n)
{
  int thread_sum = 0;
  ::cuda::for_each_canceled_cluster<1>([&](dim3 block_idx) {
    thread_sum += vec_add_impl1(a, b, c, n, block_idx);
  });
  cg::reduce_update_async(
    cg::tiled_partition<32>(cg::this_thread_block()), cuda::atomic_ref{*s}, thread_sum, cg::plus<int>{});
}

__global__ void __cluster_dims__(CLUSTER_DIM2, CLUSTER_DIM2, 1) cluster_vec_add2(int* s, int* a, int* b, int* c, int n)
{
  int thread_sum = 0;
  ::cuda::for_each_canceled_cluster<2>([&](dim3 block_idx) {
    thread_sum += vec_add_impl2(a, b, c, n, block_idx);
  });
  cg::reduce_update_async(
    cg::tiled_partition<32>(cg::this_thread_block()), cuda::atomic_ref{*s}, thread_sum, cg::plus<int>{});
}

__global__ void __cluster_dims__(CLUSTER_DIM3, CLUSTER_DIM3, CLUSTER_DIM3)
  cluster_vec_add3(int* s, int* a, int* b, int* c, int n)
{
  int thread_sum = 0;
  ::cuda::for_each_canceled_cluster<3>([&](dim3 block_idx) {
    thread_sum += vec_add_impl3(a, b, c, n, block_idx);
  });
  cg::reduce_update_async(
    cg::tiled_partition<32>(cg::this_thread_block()), cuda::atomic_ref{*s}, thread_sum, cg::plus<int>{});
}

template <typename F>
void test(F&& f)
{
  for (auto N : {1000, 10000, 1000000, 1000000})
  {
    for (int tidx : {0, 33, 63, 94})
    {
      int *s, *a, *b, *c;
      cudaMallocManaged(&s, 1 * sizeof(int));
      cudaMallocManaged(&a, N * sizeof(int));
      cudaMallocManaged(&b, N * sizeof(int));
      cudaMallocManaged(&c, N * sizeof(int));
      *s = 0;
      for (int i = 0; i < N; ++i)
      {
        a[i] = i;
        b[i] = 1;
        c[i] = 0;
      }

      f(s, a, b, c, N, tidx);
      assert(cudaGetLastError() == cudaSuccess);
      assert(cudaDeviceSynchronize() == cudaSuccess);

      int sref = 0;
      for (int i = 0; i < N; ++i)
      {
        sref += c[i];
        assert(c[i] != (1 + i));
      }
      assert(*s == sref);

      cudaFree(s);
      cudaFree(a);
      cudaFree(b);
      cudaFree(c);
    }
  }
}

void test()
{
  //////////////////////////////////////////////////////////////////////////////
  // for_each_canceled_block:

  {
    auto fn = [](int* s, int* a, int* b, int* c, int n, int tidx) {
      int tpb = 256;
      int bpg = (n + tpb - 1) / tpb;
      vec_add_det1<<<bpg, tpb>>>(s, a, b, c, n, tidx);
    };
    test(fn);
  }

  {
    auto fn = [](int* s, int* a, int* b, int* c, int n, int tidx) {
      dim3 tpb(16, 16, 1);
      int ntpb = tpb.x * tpb.y; // 256
      int bpg  = (n + ntpb - 1) / ntpb;
      int bpgx = (int) cuda::std::sqrt(bpg) + 1;
      int bpgy = bpgx;
      assert((bpgx * bpgy) >= bpg);
      vec_add_det2<<<dim3(bpgx, bpgy, 1), tpb>>>(s, a, b, c, n, tidx);
    };
    test(fn);
  }

  {
    auto fn = [](int* s, int* a, int* b, int* c, int n, int tidx) {
      dim3 tpb(8, 8, 8);
      int ntpb = tpb.x * tpb.y * tpb.z; // 512
      int bpg  = (n + ntpb - 1) / ntpb;
      int bpgx = (int) cuda::std::pow(bpg, 1. / 3.) + 1;
      int bpgy = bpgx;
      int bpgz = bpgx;
      assert((bpgx * bpgy * bpgz) >= bpg);
      vec_add_det3<<<dim3(bpgx, bpgy, bpgz), tpb>>>(s, a, b, c, n, tidx);
    };
    test(fn);
  }

  {
    auto fn = [](int* s, int* a, int* b, int* c, int n, int tidx) {
      int tpb = 256;
      int bpg = (n + tpb - 1) / tpb;
      vec_add1<<<bpg, tpb>>>(s, a, b, c, n);
    };
    test(fn);
  }

  {
    auto fn = [](int* s, int* a, int* b, int* c, int n, int tidx) {
      dim3 tpb(16, 16, 1);
      int ntpb = tpb.x * tpb.y; // 256
      int bpg  = (n + ntpb - 1) / ntpb;
      int bpgx = (int) cuda::std::sqrt(bpg) + 1;
      int bpgy = bpgx;
      assert((bpgx * bpgy) >= bpg);
      vec_add2<<<dim3(bpgx, bpgy, 1), tpb>>>(s, a, b, c, n);
    };
    test(fn);
  }

  {
    auto fn = [](int* s, int* a, int* b, int* c, int n, int tidx) {
      dim3 tpb(8, 8, 8);
      int ntpb = tpb.x * tpb.y * tpb.z; // 512
      int bpg  = (n + ntpb - 1) / ntpb;
      int bpgx = (int) cuda::std::pow(bpg, 1. / 3.) + 1;
      int bpgy = bpgx;
      int bpgz = bpgx;
      assert((bpgx * bpgy * bpgz) >= bpg);
      vec_add3<<<dim3(bpgx, bpgy, bpgz), tpb>>>(s, a, b, c, n);
    };
    test(fn);
  }

  //////////////////////////////////////////////////////////////////////////////
  // for_each_canceled_cluster:

  {
    auto fn = [](int* s, int* a, int* b, int* c, size_t n, int tidx) {
      int tpb = 256;
      int bpg = (n + tpb - 1) / tpb + 1;
      while (bpg % CLUSTER_DIM1 != 0)
      {
        bpg++;
      }
      cluster_vec_add_det1<<<bpg, tpb>>>(s, a, b, c, n, 0, tidx);
    };
    test(fn);
  }

  {
    auto fn = [](int* s, int* a, int* b, int* c, size_t n, int tidx) {
      dim3 tpb(16, 16, 1);
      int ntpb = tpb.x * tpb.y; // 256
      int bpg  = (n + ntpb - 1) / ntpb;
      int bpgx = (int) cuda::std::sqrt(bpg) + 1;
      while (bpgx % CLUSTER_DIM2 != 0)
      {
        bpgx++;
      }
      int bpgy = bpgx;
      assert((bpgx * bpgy) >= bpg);
      cluster_vec_add_det2<<<dim3(bpgx, bpgy, 1), tpb>>>(s, a, b, c, n, 0, tidx);
    };
    test(fn);
  }

  {
    auto fn = [](int* s, int* a, int* b, int* c, size_t n, int tidx) {
      dim3 tpb(8, 8, 8);
      int ntpb = tpb.x * tpb.y * tpb.z; // 512
      int bpg  = (n + ntpb - 1) / ntpb;
      int bpgx = (int) cuda::std::pow(bpg, 1. / 3.) + 1;
      while (bpgx % CLUSTER_DIM3 != 0)
      {
        bpgx++;
      }
      int bpgy = bpgx;
      int bpgz = bpgx;
      assert((bpgx * bpgy * bpgz) >= bpg);
      cluster_vec_add_det3<<<dim3(bpgx, bpgy, bpgz), tpb>>>(s, a, b, c, n, 0, tidx);
    };
    test(fn);
  }

  {
    auto fn = [](int* s, int* a, int* b, int* c, int n, int tidx) {
      int tpb = 256;
      int bpg = (n + tpb - 1) / tpb;
      while (bpg % CLUSTER_DIM1 != 0)
      {
        bpg++;
      }
      cluster_vec_add1<<<bpg, tpb>>>(s, a, b, c, n);
    };
    test(fn);
  }

  {
    auto fn = [](int* s, int* a, int* b, int* c, int n, int tidx) {
      dim3 tpb(16, 16, 1);
      int ntpb = tpb.x * tpb.y; // 256
      int bpg  = (n + ntpb - 1) / ntpb;
      int bpgx = (int) cuda::std::sqrt(bpg) + 1;
      while (bpgx % CLUSTER_DIM2 != 0)
      {
        bpgx++;
      }
      int bpgy = bpgx;
      assert((bpgx * bpgy) >= bpg);
      cluster_vec_add2<<<dim3(bpgx, bpgy, 1), tpb>>>(s, a, b, c, n);
    };
    test(fn);
  }

  {
    auto fn = [](int* s, int* a, int* b, int* c, int n, int tidx) {
      dim3 tpb(8, 8, 8);
      int ntpb = tpb.x * tpb.y * tpb.z; // 512
      int bpg  = (n + ntpb - 1) / ntpb;
      int bpgx = (int) cuda::std::pow(bpg, 1. / 3.) + 1;
      while (bpgx % CLUSTER_DIM3 != 0)
      {
        bpgx++;
      }
      int bpgy = bpgx;
      int bpgz = bpgx;
      assert((bpgx * bpgy * bpgz) >= bpg);
      cluster_vec_add3<<<dim3(bpgx, bpgy, bpgz), tpb>>>(s, a, b, c, n);
    };
    test(fn);
  }
}

#endif // !TEST_HAS_NO_INT128_T

int main(int argc, char** argv)
{
#if !defined(TEST_HAS_NO_INT128_T)
  NV_IF_TARGET(NV_IS_HOST, (test();))
#endif // !TEST_HAS_NO_INT128_T

  return 0;
}
