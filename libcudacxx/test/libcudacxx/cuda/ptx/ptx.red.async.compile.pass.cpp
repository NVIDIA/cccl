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
#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    if (non_eliminated_false()) {
      // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.inc.u32  [dest], value, [remote_bar];
      auto overload = static_cast<void (*)(cuda::ptx::op_inc_t, uint32_t* , const uint32_t& , uint64_t* )>(cuda::ptx::red_async);
      fn_ptr = reinterpret_cast<void*>(overload);
    }
  ));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    if (non_eliminated_false()) {
      // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.dec.u32  [dest], value, [remote_bar];
      auto overload = static_cast<void (*)(cuda::ptx::op_dec_t, uint32_t* , const uint32_t& , uint64_t* )>(cuda::ptx::red_async);
      fn_ptr = reinterpret_cast<void*>(overload);
    }
  ));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    if (non_eliminated_false()) {
      // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.min.u32  [dest], value, [remote_bar];
      auto overload = static_cast<void (*)(cuda::ptx::op_min_t, uint32_t* , const uint32_t& , uint64_t* )>(cuda::ptx::red_async);
      fn_ptr = reinterpret_cast<void*>(overload);
    }
  ));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    if (non_eliminated_false()) {
      // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.max.u32  [dest], value, [remote_bar];
      auto overload = static_cast<void (*)(cuda::ptx::op_max_t, uint32_t* , const uint32_t& , uint64_t* )>(cuda::ptx::red_async);
      fn_ptr = reinterpret_cast<void*>(overload);
    }
  ));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    if (non_eliminated_false()) {
      // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.add.u32  [dest], value, [remote_bar];
      auto overload = static_cast<void (*)(cuda::ptx::op_add_t, uint32_t* , const uint32_t& , uint64_t* )>(cuda::ptx::red_async);
      fn_ptr = reinterpret_cast<void*>(overload);
    }
  ));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    if (non_eliminated_false()) {
      // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.min.s32  [dest], value, [remote_bar];
      auto overload = static_cast<void (*)(cuda::ptx::op_min_t, int32_t* , const int32_t& , uint64_t* )>(cuda::ptx::red_async);
      fn_ptr = reinterpret_cast<void*>(overload);
    }
  ));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    if (non_eliminated_false()) {
      // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.max.s32  [dest], value, [remote_bar];
      auto overload = static_cast<void (*)(cuda::ptx::op_max_t, int32_t* , const int32_t& , uint64_t* )>(cuda::ptx::red_async);
      fn_ptr = reinterpret_cast<void*>(overload);
    }
  ));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    if (non_eliminated_false()) {
      // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.add.s32  [dest], value, [remote_bar];
      auto overload = static_cast<void (*)(cuda::ptx::op_add_t, int32_t* , const int32_t& , uint64_t* )>(cuda::ptx::red_async);
      fn_ptr = reinterpret_cast<void*>(overload);
    }
  ));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    if (non_eliminated_false()) {
      // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.and.b32  [dest], value, [remote_bar];
      auto overload = static_cast<void (*)(cuda::ptx::op_and_op_t, int32_t* , const int32_t& , uint64_t* )>(cuda::ptx::red_async);
      fn_ptr = reinterpret_cast<void*>(overload);
    }
  ));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    if (non_eliminated_false()) {
      // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.or.b32  [dest], value, [remote_bar];
      auto overload = static_cast<void (*)(cuda::ptx::op_or_op_t, int32_t* , const int32_t& , uint64_t* )>(cuda::ptx::red_async);
      fn_ptr = reinterpret_cast<void*>(overload);
    }
  ));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    if (non_eliminated_false()) {
      // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.xor.b32  [dest], value, [remote_bar];
      auto overload = static_cast<void (*)(cuda::ptx::op_xor_op_t, int32_t* , const int32_t& , uint64_t* )>(cuda::ptx::red_async);
      fn_ptr = reinterpret_cast<void*>(overload);
    }
  ));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    if (non_eliminated_false()) {
      // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.add.u64  [dest], value, [remote_bar];
      auto overload = static_cast<void (*)(cuda::ptx::op_add_t, uint64_t* , const uint64_t& , uint64_t* )>(cuda::ptx::red_async);
      fn_ptr = reinterpret_cast<void*>(overload);
    }
  ));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    if (non_eliminated_false()) {
      // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.add.u64  [dest], value, [remote_bar]; // .u64 intentional
      auto overload = static_cast<void (*)(cuda::ptx::op_add_t, int64_t* , const int64_t& , int64_t* )>(cuda::ptx::red_async);
      fn_ptr = reinterpret_cast<void*>(overload);
    }
  ));
#endif // __cccl_ptx_isa >= 810
}

int main(int, char**)
{
    return 0;
}
