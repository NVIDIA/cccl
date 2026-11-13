//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Mandates: is_invocable_r must fail to compile when used with device lambdas.
// UNSUPPORTED: clang && (!nvcc)

// type_traits

// is_invocable_r

#include <cuda/std/type_traits>

#include "test_macros.h"

template <class Fn>
TEST_FUNC void test_lambda(Fn&&)
{
  static_assert(cuda::std::is_invocable_r<int, Fn, int>::value);
}

int main(int, char**)
{
#if TEST_CUDA_COMPILER(NVCC) || TEST_COMPILER(NVRTC)
  { // extended device lambda
    test_lambda([] TEST_DEVICE_FUNC(int x) {
      return x;
    });
  }
#endif // TEST_CUDA_COMPILER(NVCC) || TEST_COMPILER(NVRTC)

  return 0;
}
