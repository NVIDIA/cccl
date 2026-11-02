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
// UNSUPPORTED: no_execute

// UNSUPPORTED: enable-tile
// Cluster operations are unavailable in tile code.

// ADDITIONAL_COMPILE_DEFINITIONS: CCCL_ENABLE_ASSERTIONS

#include <cuda/barrier>
#include <cuda/std/cstdint>

#include <new> // IWYU pragma: keep (needed for placement new)

#include <cooperative_groups.h>

#include "test_macros.h"

TEST_DEVICE_FUNC void test_remote_wait()
{
  alignas(cuda::shared_memory_barrier<cuda::thread_scope_cluster>)
    __shared__ char storage[sizeof(cuda::shared_memory_barrier<cuda::thread_scope_cluster>)];
  auto* bar = reinterpret_cast<cuda::shared_memory_barrier<cuda::thread_scope_cluster>*>(storage);
  new ((void*) bar) cuda::shared_memory_barrier<cuda::thread_scope_cluster>;
  init(bar, 1);

  auto cluster = cooperative_groups::this_cluster();
  cluster.sync();

  auto* remote_bar = cluster.map_shared_rank(bar, cluster.block_rank() ^ 1);
  // Even cluster-scope waits must use a barrier owned by the calling block.
  (void) remote_bar->test_wait(cuda::std::uint32_t{0}, cuda::ignore_status);

  cluster.sync();
}

int main(int, char**)
{
  NV_DISPATCH_TARGET(NV_IS_HOST, (cuda_cluster_size = 2;), NV_PROVIDES_SM_90, (test_remote_wait();))
  return 0;
}
