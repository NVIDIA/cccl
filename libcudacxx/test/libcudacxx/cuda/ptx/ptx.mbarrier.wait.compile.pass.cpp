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

/*
 * We use a special strategy to force the generation of the PTX. This is mainly
 * a fight against dead-code-elimination in the NVVM layer.
 *
 * The reason we need this strategy is because certain older versions of ptxas
 * segfault when a non-sensical sequence of PTX is generated. So instead, we try
 * to force the instantiation and compilation to PTX of all the overloads of the
 * PTX wrapping functions.
 *
 * We do this by writing a function pointer of each overload to the `__device__`
 * variable `fn_ptr`. Now, because weak stores from a single thread may be
 * elided, we also wrap the store in an if branch that cannot be removed.
 *
 * To prevent dead-code-elimination of the if branch, we use
 * `non_eliminated_false`, which uses inline assembly to hide the fact that is
 * always false from NVVM.
 *
 * So this is how we ensure that none of the function pointer stores are elided.
 * Because `fn_ptr` is possibly visible outside this translation unit, the
 * compiler must compile all the functions which are stored.
 *
 */

__device__ void * fn_ptr = nullptr;

__device__ bool non_eliminated_false(void){
  int ret = 0;
  asm ("": "=r"(ret)::);
  return ret != 0;
}

__global__ void test_compilation() {
#if __cccl_ptx_isa >= 700
  NV_IF_TARGET(NV_PROVIDES_SM_80, (
    if (non_eliminated_false()) {
      // mbarrier.test_wait.shared.b64 waitComplete, [addr], state;                                                  // 1.
      auto overload = static_cast<bool (*)(uint64_t* , const uint64_t& )>(cuda::ptx::mbarrier_test_wait);
      fn_ptr = reinterpret_cast<void*>(overload);
    }
  ));
#endif // __cccl_ptx_isa >= 700

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    if (non_eliminated_false()) {
      // mbarrier.test_wait{.sem}{.scope}.shared::cta.b64        waitComplete, [addr], state;                        // 2.
      auto overload = static_cast<bool (*)(cuda::ptx::sem_acquire_t, cuda::ptx::scope_cta_t, uint64_t* , const uint64_t& )>(cuda::ptx::mbarrier_test_wait);
      fn_ptr = reinterpret_cast<void*>(overload);
    }
    if (non_eliminated_false()) {
      // mbarrier.test_wait{.sem}{.scope}.shared::cta.b64        waitComplete, [addr], state;                        // 2.
      auto overload = static_cast<bool (*)(cuda::ptx::sem_acquire_t, cuda::ptx::scope_cluster_t, uint64_t* , const uint64_t& )>(cuda::ptx::mbarrier_test_wait);
      fn_ptr = reinterpret_cast<void*>(overload);
    }
  ));
#endif // __cccl_ptx_isa >= 800
#if __cccl_ptx_isa >= 710
  NV_IF_TARGET(NV_PROVIDES_SM_80, (
    if (non_eliminated_false()) {
      // mbarrier.test_wait.parity.shared.b64 waitComplete, [addr], phaseParity;                                     // 3.
      auto overload = static_cast<bool (*)(uint64_t* , const uint32_t& )>(cuda::ptx::mbarrier_test_wait_parity);
      fn_ptr = reinterpret_cast<void*>(overload);
    }
  ));
#endif // __cccl_ptx_isa >= 710

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    if (non_eliminated_false()) {
      // mbarrier.test_wait.parity{.sem}{.scope}.shared::cta.b64 waitComplete, [addr], phaseParity;                  // 4.
      auto overload = static_cast<bool (*)(cuda::ptx::sem_acquire_t, cuda::ptx::scope_cta_t, uint64_t* , const uint32_t& )>(cuda::ptx::mbarrier_test_wait_parity);
      fn_ptr = reinterpret_cast<void*>(overload);
    }
    if (non_eliminated_false()) {
      // mbarrier.test_wait.parity{.sem}{.scope}.shared::cta.b64 waitComplete, [addr], phaseParity;                  // 4.
      auto overload = static_cast<bool (*)(cuda::ptx::sem_acquire_t, cuda::ptx::scope_cluster_t, uint64_t* , const uint32_t& )>(cuda::ptx::mbarrier_test_wait_parity);
      fn_ptr = reinterpret_cast<void*>(overload);
    }
  ));
#endif // __cccl_ptx_isa >= 800
#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    if (non_eliminated_false()) {
      // mbarrier.try_wait.shared::cta.b64         waitComplete, [addr], state;                                      // 5a.
      auto overload = static_cast<bool (*)(uint64_t* , const uint64_t& )>(cuda::ptx::mbarrier_try_wait);
      fn_ptr = reinterpret_cast<void*>(overload);
    }
  ));
#endif // __cccl_ptx_isa >= 780

#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    if (non_eliminated_false()) {
      // mbarrier.try_wait.shared::cta.b64         waitComplete, [addr], state, suspendTimeHint;                    // 5b.
      auto overload = static_cast<bool (*)(uint64_t* , const uint64_t& , const uint32_t& )>(cuda::ptx::mbarrier_try_wait);
      fn_ptr = reinterpret_cast<void*>(overload);
    }
  ));
#endif // __cccl_ptx_isa >= 780

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    if (non_eliminated_false()) {
      // mbarrier.try_wait{.sem}{.scope}.shared::cta.b64         waitComplete, [addr], state;                        // 6a.
      auto overload = static_cast<bool (*)(cuda::ptx::sem_acquire_t, cuda::ptx::scope_cta_t, uint64_t* , const uint64_t& )>(cuda::ptx::mbarrier_try_wait);
      fn_ptr = reinterpret_cast<void*>(overload);
    }
    if (non_eliminated_false()) {
      // mbarrier.try_wait{.sem}{.scope}.shared::cta.b64         waitComplete, [addr], state;                        // 6a.
      auto overload = static_cast<bool (*)(cuda::ptx::sem_acquire_t, cuda::ptx::scope_cluster_t, uint64_t* , const uint64_t& )>(cuda::ptx::mbarrier_try_wait);
      fn_ptr = reinterpret_cast<void*>(overload);
    }
  ));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    if (non_eliminated_false()) {
      // mbarrier.try_wait{.sem}{.scope}.shared::cta.b64         waitComplete, [addr], state , suspendTimeHint;      // 6b.
      auto overload = static_cast<bool (*)(cuda::ptx::sem_acquire_t, cuda::ptx::scope_cta_t, uint64_t* , const uint64_t& , const uint32_t& )>(cuda::ptx::mbarrier_try_wait);
      fn_ptr = reinterpret_cast<void*>(overload);
    }
    if (non_eliminated_false()) {
      // mbarrier.try_wait{.sem}{.scope}.shared::cta.b64         waitComplete, [addr], state , suspendTimeHint;      // 6b.
      auto overload = static_cast<bool (*)(cuda::ptx::sem_acquire_t, cuda::ptx::scope_cluster_t, uint64_t* , const uint64_t& , const uint32_t& )>(cuda::ptx::mbarrier_try_wait);
      fn_ptr = reinterpret_cast<void*>(overload);
    }
  ));
#endif // __cccl_ptx_isa >= 800
#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    if (non_eliminated_false()) {
      // mbarrier.try_wait.parity.shared::cta.b64  waitComplete, [addr], phaseParity;                                // 7a.
      auto overload = static_cast<bool (*)(uint64_t* , const uint32_t& )>(cuda::ptx::mbarrier_try_wait_parity);
      fn_ptr = reinterpret_cast<void*>(overload);
    }
  ));
#endif // __cccl_ptx_isa >= 780

#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    if (non_eliminated_false()) {
      // mbarrier.try_wait.parity.shared::cta.b64  waitComplete, [addr], phaseParity, suspendTimeHint;               // 7b.
      auto overload = static_cast<bool (*)(uint64_t* , const uint32_t& , const uint32_t& )>(cuda::ptx::mbarrier_try_wait_parity);
      fn_ptr = reinterpret_cast<void*>(overload);
    }
  ));
#endif // __cccl_ptx_isa >= 780

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    if (non_eliminated_false()) {
      // mbarrier.try_wait.parity{.sem}{.scope}.shared::cta.b64  waitComplete, [addr], phaseParity;                  // 8a.
      auto overload = static_cast<bool (*)(cuda::ptx::sem_acquire_t, cuda::ptx::scope_cta_t, uint64_t* , const uint32_t& )>(cuda::ptx::mbarrier_try_wait_parity);
      fn_ptr = reinterpret_cast<void*>(overload);
    }
    if (non_eliminated_false()) {
      // mbarrier.try_wait.parity{.sem}{.scope}.shared::cta.b64  waitComplete, [addr], phaseParity;                  // 8a.
      auto overload = static_cast<bool (*)(cuda::ptx::sem_acquire_t, cuda::ptx::scope_cluster_t, uint64_t* , const uint32_t& )>(cuda::ptx::mbarrier_try_wait_parity);
      fn_ptr = reinterpret_cast<void*>(overload);
    }
  ));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    if (non_eliminated_false()) {
      // mbarrier.try_wait.parity{.sem}{.scope}.shared::cta.b64  waitComplete, [addr], phaseParity, suspendTimeHint; // 8b.
      auto overload = static_cast<bool (*)(cuda::ptx::sem_acquire_t, cuda::ptx::scope_cta_t, uint64_t* , const uint32_t& , const uint32_t& )>(cuda::ptx::mbarrier_try_wait_parity);
      fn_ptr = reinterpret_cast<void*>(overload);
    }
    if (non_eliminated_false()) {
      // mbarrier.try_wait.parity{.sem}{.scope}.shared::cta.b64  waitComplete, [addr], phaseParity, suspendTimeHint; // 8b.
      auto overload = static_cast<bool (*)(cuda::ptx::sem_acquire_t, cuda::ptx::scope_cluster_t, uint64_t* , const uint32_t& , const uint32_t& )>(cuda::ptx::mbarrier_try_wait_parity);
      fn_ptr = reinterpret_cast<void*>(overload);
    }
  ));
#endif // __cccl_ptx_isa >= 800
}

int main(int, char**)
{
    return 0;
}
