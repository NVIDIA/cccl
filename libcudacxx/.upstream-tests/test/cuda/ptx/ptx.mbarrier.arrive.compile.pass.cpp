//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: libcpp-has-no-threads

// <cuda/ptx>

#include <cuda/ptx>
#include <cuda/std/utility>

#include "concurrent_agents.h"
#include "cuda_space_selector.h"
#include "test_macros.h"

template <typename ... _Ty>
__device__ inline bool __unused(_Ty...) { return true; }

__global__ void test_compilation() {
  using cuda::ptx::sem_release;
  using cuda::ptx::space_cluster;
  using cuda::ptx::space_shared;
  using cuda::ptx::scope_cluster;
  using cuda::ptx::scope_cta;

  __shared__ uint64_t bar;
  bar = 1;
  uint64_t state = 1;

#if __cccl_ptx_isa >= 700
  NV_IF_TARGET(NV_PROVIDES_SM_80, (
    state = cuda::ptx::mbarrier_arrive(&bar);                                                        // 1.
    state = cuda::ptx::mbarrier_arrive_no_complete(&bar, 1);                                         // 5.
  ));
#endif // __cccl_ptx_isa >= 700

  // This guard is redundant: before PTX ISA 7.8, there was no support for SM_90
#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    state = cuda::ptx::mbarrier_arrive(&bar, 1);                                                     // 2.
  ));
#endif // __cccl_ptx_isa >= 780

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    state = cuda::ptx::mbarrier_arrive(sem_release, scope_cta, space_shared, &bar);                  // 3a.
    state = cuda::ptx::mbarrier_arrive(sem_release, scope_cluster, space_shared, &bar);              // 3a.

    state = cuda::ptx::mbarrier_arrive(sem_release, scope_cta, space_shared, &bar, 1);               // 3b.
    state = cuda::ptx::mbarrier_arrive(sem_release, scope_cluster, space_shared, &bar, 1);           // 3b.

    cuda::ptx::mbarrier_arrive(sem_release, scope_cluster, space_cluster, &bar);                     // 4a.

    cuda::ptx::mbarrier_arrive(sem_release, scope_cluster, space_cluster, &bar, 1);                  // 4b.

    state = cuda::ptx::mbarrier_arrive_expect_tx(sem_release, scope_cta, space_shared, &bar, 1);     // 8.
    state = cuda::ptx::mbarrier_arrive_expect_tx(sem_release, scope_cluster, space_shared, &bar, 1); // 8.

    cuda::ptx::mbarrier_arrive_expect_tx(sem_release, scope_cluster, space_cluster, &bar, 1);        // 9.
  ));
#endif // __cccl_ptx_isa >= 800
  __unused(bar, state);
}

int main(int, char**)
{
    return 0;
}
