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

int main(int, char**)
{
    NV_IF_TARGET(NV_IS_DEVICE, (
        // Do not execute. Just check if below PTX compiles (that is: assembles) without error.

        // This condition always evaluates to false, but the compiler does not
        // reason through it. This avoids dead code elimination.
        const bool non_eliminated_false = threadIdx.x > 1024;

        if (non_eliminated_false) {
            using cuda::ptx::sem_release;
            using cuda::ptx::space_shared_cluster;
            using cuda::ptx::space_shared;
            using cuda::ptx::scope_cluster;
            using cuda::ptx::scope_cta;

            __shared__ uint64_t bar;
            bar = 1;
            uint64_t state = 1;

            NV_IF_TARGET(NV_PROVIDES_SM_80, (
#if __cccl_ptx_isa >= 700
              state = cuda::ptx::mbarrier_arrive(sem_release, scope_cta,     space_shared, &bar);              // 1.
              state = cuda::ptx::mbarrier_arrive_no_complete(sem_release, scope_cta, space_shared, &bar, 1);   // 2.
#endif
            ));

            NV_IF_TARGET(NV_PROVIDES_SM_90, (
#if __cccl_ptx_isa >= 780 // This guard is redundant: before PTX ISA 7.8, there was no support for SM_90
              state = cuda::ptx::mbarrier_arrive(sem_release, scope_cta,     space_shared, &bar, 1);           // 3.
#endif
#if __cccl_ptx_isa >= 800
              state = cuda::ptx::mbarrier_arrive(sem_release, scope_cluster, space_shared, &bar, 1);           // 4.

              cuda::ptx::mbarrier_arrive(sem_release, scope_cta,     space_shared_cluster, &bar, 1);           // 5.
              cuda::ptx::mbarrier_arrive(sem_release, scope_cluster, space_shared_cluster, &bar, 1);           // 5.

              state = cuda::ptx::mbarrier_arrive_expect_tx(sem_release, scope_cta,     space_shared, &bar, 1); // 6.
              state = cuda::ptx::mbarrier_arrive_expect_tx(sem_release, scope_cluster, space_shared, &bar, 1); // 6.

              cuda::ptx::mbarrier_arrive_expect_tx(sem_release, scope_cta,     space_shared_cluster, &bar, 1); // 7.
              cuda::ptx::mbarrier_arrive_expect_tx(sem_release, scope_cluster, space_shared_cluster, &bar, 1); // 7.
#endif
            ));

            __unused(bar, state);
        }
    ));

    return 0;
}
