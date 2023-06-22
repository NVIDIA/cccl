//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: pre-sm-80

#pragma nv_diag_suppress static_var_with_dynamic_init
#pragma nv_diag_suppress set_but_not_used

#include <cuda/barrier>

#include "cuda_space_selector.h"

__device__
void test()
{
    __shared__ cuda::barrier<cuda::thread_scope_block> b;
    init(&b, 2);

    uint64_t token;
    asm volatile ("mbarrier.arrive.b64 %0, [%1];"
        : "=l"(token)
        : "l"(cuda::device::barrier_native_handle(b))
        : "memory");
    (void)token;

    b.arrive_and_wait();
}

int main(int argc, char ** argv)
{
    NV_IF_TARGET(NV_PROVIDES_SM_80,
        test();
    )

    return 0;
}
