//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: pre-sm-70

#ifndef __NVCOMPILER
#pragma nv_diag_suppress static_var_with_dynamic_init
#endif
#pragma nv_diag_suppress set_but_not_used

#include <cuda/barrier>

int main(int argc, char ** argv)
{
#if __CUDA_ARCH__ >= 800
    __shared__ cuda::barrier<cuda::thread_scope_block> b;
    init(&b, 2);

    uint64_t token;
    asm volatile ("mbarrier.arrive.b64 %0, [%1];"
        : "=l"(token)
        : "l"(cuda::device::barrier_native_handle(b))
        : "memory");
    (void)token;

    b.arrive_and_wait();
#endif

    return 0;
}
