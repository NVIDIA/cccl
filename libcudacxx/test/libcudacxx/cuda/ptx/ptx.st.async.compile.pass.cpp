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
// UNSUPPORTED: c++11

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
      // st.async.weak.shared::cluster.mbarrier::complete_tx::bytes{.type} [addr], value, [remote_bar];    // 1.
      auto overload = static_cast<void (*)(int32_t* , const int32_t& , uint64_t* )>(cuda::ptx::st_async);
      fn_ptr = reinterpret_cast<void*>(overload);
    }
    if (non_eliminated_false()) {
      // st.async.weak.shared::cluster.mbarrier::complete_tx::bytes{.type} [addr], value, [remote_bar];    // 1.
      auto overload = static_cast<void (*)(int64_t* , const int64_t& , uint64_t* )>(cuda::ptx::st_async);
      fn_ptr = reinterpret_cast<void*>(overload);
    }
  ));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    if (non_eliminated_false()) {
      // st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v2{.type} [addr], value, [remote_bar]; // 2.
      auto overload = static_cast<void (*)(int32_t* , const int32_t (&)[2], uint64_t* )>(cuda::ptx::st_async);
      fn_ptr = reinterpret_cast<void*>(overload);
    }
    if (non_eliminated_false()) {
      // st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v2{.type} [addr], value, [remote_bar]; // 2.
      auto overload = static_cast<void (*)(int64_t* , const int64_t (&)[2], uint64_t* )>(cuda::ptx::st_async);
      fn_ptr = reinterpret_cast<void*>(overload);
    }
  ));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(NV_PROVIDES_SM_90, (
    if (non_eliminated_false()) {
      // st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [addr], value, [remote_bar];    // 3.
      auto overload = static_cast<void (*)(int32_t* , const int32_t (&)[4], uint64_t* )>(cuda::ptx::st_async);
      fn_ptr = reinterpret_cast<void*>(overload);
    }
  ));
#endif // __cccl_ptx_isa >= 810
}

int main(int, char**)
{
    return 0;
}
