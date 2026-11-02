//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc
// UNSUPPORTED: nvcc-12, nvcc-13.0, nvcc-13.1, nvcc-13.2, nvcc-13.3
// UNSUPPORTED: pre-sm-90

// UNSUPPORTED: enable-tile
// Cluster operations are unavailable in tile code.

#include <cuda/barrier>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <new> // IWYU pragma: keep (needed for placement new)

#include <cooperative_groups.h>

#include "test_macros.h"

static_assert(cuda::std::is_same_v<
              decltype(cuda::std::declval<cuda::shared_memory_barrier<cuda::thread_scope_cluster>&>().remote_arrive()),
              void>);
static_assert(
  cuda::std::is_same_v<
    decltype(cuda::std::declval<cuda::shared_memory_barrier<cuda::thread_scope_cluster>&>().remote_arrive_tx(1, 0)),
    void>);
static_assert(
  cuda::std::is_same_v<
    decltype(cuda::std::declval<cuda::shared_memory_barrier<cuda::thread_scope_cluster>&>().remote_expect_tx(0)),
    void>);
static_assert(
  cuda::std::is_same_v<
    decltype(cuda::std::declval<cuda::shared_memory_barrier<cuda::thread_scope_cluster>&>().remote_complete_tx(0)),
    void>);

template <int Id, bool WithTransactions, bool SeparateExpectation = false>
TEST_DEVICE_FUNC void test_remote_arrive(int arrivals, int transactions = 0)
{
  alignas(cuda::shared_memory_barrier<cuda::thread_scope_cluster>)
    __shared__ char storage[sizeof(cuda::shared_memory_barrier<cuda::thread_scope_cluster>)];
  __shared__ int value;
  auto* bar    = reinterpret_cast<cuda::shared_memory_barrier<cuda::thread_scope_cluster>*>(storage);
  auto cluster = cooperative_groups::this_cluster();

  if (cluster.block_rank() == 0)
  {
    new ((void*) bar) cuda::shared_memory_barrier<cuda::thread_scope_cluster>;
    init(bar, arrivals);
    value = 0;
  }
  cluster.sync();

  if (cluster.block_rank() == 1)
  {
    auto* remote_bar                    = cluster.map_shared_rank(bar, 0);
    *cluster.map_shared_rank(&value, 0) = 42;
    if constexpr (WithTransactions)
    {
      if constexpr (SeparateExpectation)
      {
        remote_bar->remote_expect_tx(transactions);
        remote_bar->remote_arrive(arrivals);
      }
      else
      {
        remote_bar->remote_arrive_tx(arrivals, transactions);
      }
      if (transactions != 0)
      {
        remote_bar->remote_complete_tx(transactions);
      }
    }
    else if (arrivals == 1)
    {
      remote_bar->remote_arrive();
    }
    else
    {
      remote_bar->remote_arrive(arrivals);
    }
  }
  else
  {
    bar->wait(cuda::std::uint32_t{0}, cuda::ignore_status);
    assert(value == 42);
  }

  cluster.sync();
  if (cluster.block_rank() == 0)
  {
    bar->~shared_memory_barrier();
  }
}

int main(int, char**)
{
  NV_DISPATCH_TARGET(
    NV_IS_HOST,
    (int device = 0; int supports_clusters = 0; assert(cudaGetDevice(&device) == cudaSuccess);
     assert(cudaDeviceGetAttribute(&supports_clusters, cudaDevAttrClusterLaunch, device) == cudaSuccess);
     cuda_cluster_size = supports_clusters ? 2 : 1;),
    NV_PROVIDES_SM_90,
    (if (cooperative_groups::this_cluster().num_blocks() == 2) {
      test_remote_arrive<0, false>(1);
      test_remote_arrive<1, false>(3);
      test_remote_arrive<2, true>(1, 0);
      test_remote_arrive<3, true>(1, 16);
      test_remote_arrive<4, true>(3, 16);
      test_remote_arrive<5, true, true>(1, 16);
      test_remote_arrive<6, true, true>(3, 16);
      test_remote_arrive<7, true, true>(1, 0);
    }))
  return 0;
}
