//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: pre-sm-70

#include "group_memcpy_async.h"

int main(int argc, char ** argv)
{
#ifndef __CUDA_ARCH__
    cuda_thread_count = 4;
#endif

    test_select_source<storage<int32_t>>();

    return 0;
}
