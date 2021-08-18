//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: pre-sm-70
// UNSUPPORTED: c++98, c++03

#include "annotated_ptr.h"

static_assert(sizeof(cuda::annotated_ptr<int, cuda::access_property::shared>) == sizeof(uintptr_t), "annotated_ptr<T, shared> must be pointer size");
static_assert(sizeof(cuda::annotated_ptr<char, cuda::access_property::shared>) == sizeof(uintptr_t), "annotated_ptr<T, shared> must be pointer size");
static_assert(sizeof(cuda::annotated_ptr<uintptr_t, cuda::access_property::shared>) == sizeof(uintptr_t), "annotated_ptr<T, shared> must be pointer size");
static_assert(sizeof(cuda::annotated_ptr<int, cuda::access_property::global>) == sizeof(uintptr_t), "annotated_ptr<T, global> must be pointer size");
static_assert(sizeof(cuda::annotated_ptr<char, cuda::access_property::global>) == sizeof(uintptr_t), "annotated_ptr<T, global> must be pointer size");
static_assert(sizeof(cuda::annotated_ptr<uintptr_t, cuda::access_property::global>) == sizeof(uintptr_t), "annotated_ptr<T, global> must be pointer size");
static_assert(sizeof(cuda::annotated_ptr<uintptr_t, cuda::access_property>) == 2*sizeof(uintptr_t), "annotated_ptr<T,access_property> must be 2 * pointer size");

static_assert(alignof(cuda::annotated_ptr<int, cuda::access_property::shared>) == alignof(int*), "annotated_ptr must align with int*");
static_assert(alignof(cuda::annotated_ptr<int, cuda::access_property::global>) == alignof(int*), "annotated_ptr must align with int*");
static_assert(alignof(cuda::annotated_ptr<int, cuda::access_property>) == alignof(int*), "annotated_ptr must align with int*");

int main(int argc, char ** argv)
{
    test_access_property_interleave();
    test_access_property_block();
    test_access_property_functions();
    test_annotated_ptr_basic();
    test_annotated_ptr_launch_kernel();
    test_annotated_ptr_functions();

    return 0;
}
