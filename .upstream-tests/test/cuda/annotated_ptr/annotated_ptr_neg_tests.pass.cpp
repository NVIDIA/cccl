//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: pre-sm-70
// UNSUPPORTED: !nvcc
// UNSUPPORTED: nvrtc
// XFAIL: nvcc

#include "utils.h"

__device__ __host__
static void fails_from_host() {
    int a;
    __nv_associate_access_property(&a, uint64_t{0});
}

int main(int argc, char ** argv) {
    //calling from host needs to fail and kill the app
    fails_from_host();
    return 0;
}
