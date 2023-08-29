//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: pre-sm-70

// <cuda/barrier>

#include <cuda/barrier>

#if defined(__CUDA_MINIMUM_ARCH__) && 700 <= __CUDA_MINIMUM_ARCH__

#ifndef  __cccl_lib_local_barrier_arrive_tx
static_assert(false, "should define __cccl_lib_local_barrier_arrive_tx");
#endif // __cccl_lib_local_barrier_arrive_tx

#endif // __CUDA_MINIMUM_ARCH__

int main(int, char**){
    return 0;
}
