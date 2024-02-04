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
      // mbarrier.arrive.shared.b64                                  state,  [addr];           // 1.
      auto overload = static_cast<uint64_t (*)(uint64_t* )>(cuda::ptx::mbarrier_arrive);
      fn_ptr = reinterpret_cast<void*>(overload);
    }
  ));
#endif // __cccl_ptx_isa >= 700

#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    if (non_eliminated_false()) {
      // mbarrier.arrive.shared::cta.b64                             state,  [addr], count;    // 2.
      auto overload = static_cast<uint64_t (*)(uint64_t* , const uint32_t& )>(cuda::ptx::mbarrier_arrive);
      fn_ptr = reinterpret_cast<void*>(overload);
    }
  ));
#endif // __cccl_ptx_isa >= 780

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    if (non_eliminated_false()) {
      // mbarrier.arrive.release.cta.shared.b64                   state,  [addr];           // 3a.
      auto overload = static_cast<uint64_t (*)(cuda::ptx::sem_release_t, cuda::ptx::scope_cta_t, cuda::ptx::space_shared_t, uint64_t* )>(cuda::ptx::mbarrier_arrive);
      fn_ptr = reinterpret_cast<void*>(overload);
    }
    if (non_eliminated_false()) {
      // mbarrier.arrive.release.cluster.shared.b64                   state,  [addr];           // 3a.
      auto overload = static_cast<uint64_t (*)(cuda::ptx::sem_release_t, cuda::ptx::scope_cluster_t, cuda::ptx::space_shared_t, uint64_t* )>(cuda::ptx::mbarrier_arrive);
      fn_ptr = reinterpret_cast<void*>(overload);
    }
  ));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    if (non_eliminated_false()) {
      // mbarrier.arrive.release.cta.shared.b64                   state,  [addr], count;    // 3b.
      auto overload = static_cast<uint64_t (*)(cuda::ptx::sem_release_t, cuda::ptx::scope_cta_t, cuda::ptx::space_shared_t, uint64_t* , const uint32_t& )>(cuda::ptx::mbarrier_arrive);
      fn_ptr = reinterpret_cast<void*>(overload);
    }
    if (non_eliminated_false()) {
      // mbarrier.arrive.release.cluster.shared.b64                   state,  [addr], count;    // 3b.
      auto overload = static_cast<uint64_t (*)(cuda::ptx::sem_release_t, cuda::ptx::scope_cluster_t, cuda::ptx::space_shared_t, uint64_t* , const uint32_t& )>(cuda::ptx::mbarrier_arrive);
      fn_ptr = reinterpret_cast<void*>(overload);
    }
  ));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    if (non_eliminated_false()) {
      // mbarrier.arrive.release.cluster.shared::cluster.b64                   _, [addr];                // 4a.
      auto overload = static_cast<void (*)(cuda::ptx::sem_release_t, cuda::ptx::scope_cluster_t, cuda::ptx::space_cluster_t, uint64_t* )>(cuda::ptx::mbarrier_arrive);
      fn_ptr = reinterpret_cast<void*>(overload);
    }
  ));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    if (non_eliminated_false()) {
      // mbarrier.arrive.release.cluster.shared::cluster.b64                   _, [addr], count;         // 4b.
      auto overload = static_cast<void (*)(cuda::ptx::sem_release_t, cuda::ptx::scope_cluster_t, cuda::ptx::space_cluster_t, uint64_t* , const uint32_t& )>(cuda::ptx::mbarrier_arrive);
      fn_ptr = reinterpret_cast<void*>(overload);
    }
  ));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 700
  NV_IF_TARGET(NV_PROVIDES_SM_80, (
    if (non_eliminated_false()) {
      // mbarrier.arrive.noComplete.shared.b64                       state,  [addr], count;    // 5.
      auto overload = static_cast<uint64_t (*)(uint64_t* , const uint32_t& )>(cuda::ptx::mbarrier_arrive_no_complete);
      fn_ptr = reinterpret_cast<void*>(overload);
    }
  ));
#endif // __cccl_ptx_isa >= 700

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    if (non_eliminated_false()) {
      // mbarrier.arrive.expect_tx.release.cta.shared.b64 state, [addr], tx_count; // 8.
      auto overload = static_cast<uint64_t (*)(cuda::ptx::sem_release_t, cuda::ptx::scope_cta_t, cuda::ptx::space_shared_t, uint64_t* , const uint32_t& )>(cuda::ptx::mbarrier_arrive_expect_tx);
      fn_ptr = reinterpret_cast<void*>(overload);
    }
    if (non_eliminated_false()) {
      // mbarrier.arrive.expect_tx.release.cluster.shared.b64 state, [addr], tx_count; // 8.
      auto overload = static_cast<uint64_t (*)(cuda::ptx::sem_release_t, cuda::ptx::scope_cluster_t, cuda::ptx::space_shared_t, uint64_t* , const uint32_t& )>(cuda::ptx::mbarrier_arrive_expect_tx);
      fn_ptr = reinterpret_cast<void*>(overload);
    }
  ));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    if (non_eliminated_false()) {
      // mbarrier.arrive.expect_tx.release.cluster.shared::cluster.b64   _, [addr], tx_count; // 9.
      auto overload = static_cast<void (*)(cuda::ptx::sem_release_t, cuda::ptx::scope_cluster_t, cuda::ptx::space_cluster_t, uint64_t* , const uint32_t& )>(cuda::ptx::mbarrier_arrive_expect_tx);
      fn_ptr = reinterpret_cast<void*>(overload);
    }
  ));
#endif // __cccl_ptx_isa >= 800
}

int main(int, char**)
{
    return 0;
}
