//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: pre-sm-70

#pragma nv_diag_suppress static_var_with_dynamic_init
#pragma nv_diag_suppress set_but_not_used

#include <cuda/barrier>

#include "cuda_space_selector.h"

__device__
void test()
{
    shared_memory_selector<cuda::barrier<cuda::thread_scope_block>, constructor_initializer> sel;
    SHARED cuda::barrier<cuda::thread_scope_block>* b;
    b = sel.construct();
    init(b, 2);

    uint64_t token;
    asm volatile ("mbarrier.arrive.b64 %0, [%1];"
        : "=l"(token)
        : "l"(cuda::device::barrier_native_handle(*b))
        : "memory");
    (void)token;

    b->arrive_and_wait();
}

int main(int argc, char ** argv)
{
#if __CUDA_ARCH__ >= 800
  test();
#endif

    return 0;
}
