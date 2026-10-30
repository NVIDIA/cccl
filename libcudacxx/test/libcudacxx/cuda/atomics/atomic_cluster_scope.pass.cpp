//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcpp-has-no-threads, pre-sm-90
// UNSUPPORTED: clang && !nvcc

// UNSUPPORTED: no_execute

// UNSUPPORTED: force-tile
// error: asm statement is unsupported in tile code

// <cuda/atomic>

#include <cuda/atomic>
#include <cuda/std/cassert>

#include <cooperative_groups.h>

#include "test_macros.h"

__device__ cuda::atomic<int, cuda::thread_scope_cluster> owning_atomic{0};

__device__ void test_cluster_scope()
{
  namespace cg       = cooperative_groups;
  const auto cluster = cg::this_cluster();

  if (cluster.block_rank() == 0)
  {
    owning_atomic.store(0, cuda::std::memory_order_relaxed);
  }
  cluster.sync();

  owning_atomic.fetch_add(1, cuda::std::memory_order_relaxed);
  cuda::atomic_thread_fence(cuda::std::memory_order_seq_cst, cuda::thread_scope_cluster);
  cluster.sync();
  assert(owning_atomic.load(cuda::std::memory_order_relaxed) == 2);

  __shared__ int shared_value;
  if (cluster.block_rank() == 0)
  {
    shared_value = 0;
  }
  cluster.sync();

  int* const remote_value = cluster.map_shared_rank(&shared_value, 0);
  cuda::atomic_ref<int, cuda::thread_scope_cluster>{*remote_value}.fetch_add(1, cuda::std::memory_order_relaxed);
  cluster.sync();
  assert(*remote_value == 2);
}

int main(int, char**)
{
  NV_DISPATCH_TARGET(NV_IS_HOST, (cuda_cluster_size = 2;), NV_PROVIDES_SM_90, (test_cluster_scope();))

  return 0;
}
