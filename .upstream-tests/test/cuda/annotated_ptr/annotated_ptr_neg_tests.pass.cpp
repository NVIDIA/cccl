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

#include "annotated_ptr.h"

__device__ __host__ __noinline__
static void negative_test_access_property_functions() {
    size_t ARR_SZ = 1 << 20;
    int* arr1 = nullptr;
    cuda::access_property ap(cuda::access_property::persisting{});

    arr1 = (int*)malloc(ARR_SZ * sizeof(int));

    //calling from host needs to fail and kill the app
    __nv_associate_access_property(arr1, static_cast<uint64_t>(ap));
}

int main(int argc, char ** argv) {
    negative_test_access_property_functions();

    return 0;
}
