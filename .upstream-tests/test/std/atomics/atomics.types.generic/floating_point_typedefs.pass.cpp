//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70

// <cuda/std/atomic>

// typedef atomic<float>   atomic_float;
// typedef atomic<double>  atomic_double;

#include <cuda/std/atomic>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
    static_assert((cuda::std::is_same<cuda::std::atomic<float>, cuda::std::atomic_float>::value), "");
    static_assert((cuda::std::is_same<cuda::std::atomic<double>, cuda::std::atomic_double>::value), "");

  return 0;
}
