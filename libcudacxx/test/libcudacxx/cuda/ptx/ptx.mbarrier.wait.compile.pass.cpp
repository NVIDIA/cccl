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
  uint64_t * addr = &bar;
  uint64_t state = 1;
  uint32_t phaseParity = 1;
  uint32_t suspendTimeHint = 1;
  bool waitComplete = true;

  int thread_filter = 1024;

#if __cccl_ptx_isa >= 700
  NV_IF_TARGET(NV_PROVIDES_SM_80, (
    if (threadIdx.x > thread_filter++) {
      // mbarrier.test_wait.shared.b64 waitComplete, [addr], state;                                                  // 1.
      waitComplete = cuda::ptx::mbarrier_test_wait(addr, state);
    }
  ));
#endif // __cccl_ptx_isa >= 700

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    if (threadIdx.x > thread_filter++) {
      // mbarrier.test_wait{.sem}{.scope}.shared::cta.b64        waitComplete, [addr], state;                        // 2.
      waitComplete = cuda::ptx::mbarrier_test_wait(cuda::ptx::sem_acquire, cuda::ptx::scope_cta, addr, state);
    }
    if (threadIdx.x > thread_filter++) {
      // mbarrier.test_wait{.sem}{.scope}.shared::cta.b64        waitComplete, [addr], state;                        // 2.
      waitComplete = cuda::ptx::mbarrier_test_wait(cuda::ptx::sem_acquire, cuda::ptx::scope_cluster, addr, state);
    }
  ));
#endif // __cccl_ptx_isa >= 800
#if __cccl_ptx_isa >= 710
  NV_IF_TARGET(NV_PROVIDES_SM_80, (
    if (threadIdx.x > thread_filter++) {
      // mbarrier.test_wait.parity.shared.b64 waitComplete, [addr], phaseParity;                                     // 3.
      waitComplete = cuda::ptx::mbarrier_test_wait_parity(addr, phaseParity);
    }
  ));
#endif // __cccl_ptx_isa >= 710

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    if (threadIdx.x > thread_filter++) {
      // mbarrier.test_wait.parity{.sem}{.scope}.shared::cta.b64 waitComplete, [addr], phaseParity;                  // 4.
      waitComplete = cuda::ptx::mbarrier_test_wait_parity(cuda::ptx::sem_acquire, cuda::ptx::scope_cta, addr, phaseParity);
    }
    if (threadIdx.x > thread_filter++) {
      // mbarrier.test_wait.parity{.sem}{.scope}.shared::cta.b64 waitComplete, [addr], phaseParity;                  // 4.
      waitComplete = cuda::ptx::mbarrier_test_wait_parity(cuda::ptx::sem_acquire, cuda::ptx::scope_cluster, addr, phaseParity);
    }
  ));
#endif // __cccl_ptx_isa >= 800
#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    if (threadIdx.x > thread_filter++) {
      // mbarrier.try_wait.shared::cta.b64         waitComplete, [addr], state;                                      // 5a.
      waitComplete = cuda::ptx::mbarrier_try_wait(addr, state);
    }
  ));
#endif // __cccl_ptx_isa >= 780

#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    if (threadIdx.x > thread_filter++) {
      // mbarrier.try_wait.shared::cta.b64         waitComplete, [addr], state, suspendTimeHint;                    // 5b.
      waitComplete = cuda::ptx::mbarrier_try_wait(addr, state, suspendTimeHint);
    }
  ));
#endif // __cccl_ptx_isa >= 780

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    if (threadIdx.x > thread_filter++) {
      // mbarrier.try_wait{.sem}{.scope}.shared::cta.b64         waitComplete, [addr], state;                        // 6a.
      waitComplete = cuda::ptx::mbarrier_try_wait(cuda::ptx::sem_acquire, cuda::ptx::scope_cta, addr, state);
    }
    if (threadIdx.x > thread_filter++) {
      // mbarrier.try_wait{.sem}{.scope}.shared::cta.b64         waitComplete, [addr], state;                        // 6a.
      waitComplete = cuda::ptx::mbarrier_try_wait(cuda::ptx::sem_acquire, cuda::ptx::scope_cluster, addr, state);
    }
  ));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    if (threadIdx.x > thread_filter++) {
      // mbarrier.try_wait{.sem}{.scope}.shared::cta.b64         waitComplete, [addr], state , suspendTimeHint;      // 6b.
      waitComplete = cuda::ptx::mbarrier_try_wait(cuda::ptx::sem_acquire, cuda::ptx::scope_cta, addr, state, suspendTimeHint);
    }
    if (threadIdx.x > thread_filter++) {
      // mbarrier.try_wait{.sem}{.scope}.shared::cta.b64         waitComplete, [addr], state , suspendTimeHint;      // 6b.
      waitComplete = cuda::ptx::mbarrier_try_wait(cuda::ptx::sem_acquire, cuda::ptx::scope_cluster, addr, state, suspendTimeHint);
    }
  ));
#endif // __cccl_ptx_isa >= 800
#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    if (threadIdx.x > thread_filter++) {
      // mbarrier.try_wait.parity.shared::cta.b64  waitComplete, [addr], phaseParity;                                // 7a.
      waitComplete = cuda::ptx::mbarrier_try_wait_parity(addr, phaseParity);
    }
  ));
#endif // __cccl_ptx_isa >= 780

#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    if (threadIdx.x > thread_filter++) {
      // mbarrier.try_wait.parity.shared::cta.b64  waitComplete, [addr], phaseParity, suspendTimeHint;               // 7b.
      waitComplete = cuda::ptx::mbarrier_try_wait_parity(addr, phaseParity, suspendTimeHint);
    }
  ));
#endif // __cccl_ptx_isa >= 780

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    if (threadIdx.x > thread_filter++) {
      // mbarrier.try_wait.parity{.sem}{.scope}.shared::cta.b64  waitComplete, [addr], phaseParity;                  // 8a.
      waitComplete = cuda::ptx::mbarrier_try_wait_parity(cuda::ptx::sem_acquire, cuda::ptx::scope_cta, addr, phaseParity);
    }
    if (threadIdx.x > thread_filter++) {
      // mbarrier.try_wait.parity{.sem}{.scope}.shared::cta.b64  waitComplete, [addr], phaseParity;                  // 8a.
      waitComplete = cuda::ptx::mbarrier_try_wait_parity(cuda::ptx::sem_acquire, cuda::ptx::scope_cluster, addr, phaseParity);
    }
  ));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    if (threadIdx.x > thread_filter++) {
      // mbarrier.try_wait.parity{.sem}{.scope}.shared::cta.b64  waitComplete, [addr], phaseParity, suspendTimeHint; // 8b.
      waitComplete = cuda::ptx::mbarrier_try_wait_parity(cuda::ptx::sem_acquire, cuda::ptx::scope_cta, addr, phaseParity, suspendTimeHint);
    }
    if (threadIdx.x > thread_filter++) {
      // mbarrier.try_wait.parity{.sem}{.scope}.shared::cta.b64  waitComplete, [addr], phaseParity, suspendTimeHint; // 8b.
      waitComplete = cuda::ptx::mbarrier_try_wait_parity(cuda::ptx::sem_acquire, cuda::ptx::scope_cluster, addr, phaseParity, suspendTimeHint);
    }
  ));
#endif // __cccl_ptx_isa >= 800

  __unused(bar, addr, state, waitComplete, phaseParity, suspendTimeHint);
}

int main(int, char**)
{
    return 0;
}
