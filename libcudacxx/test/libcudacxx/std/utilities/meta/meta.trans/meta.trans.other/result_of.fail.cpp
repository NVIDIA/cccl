//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Mandates: invoke result must fail to compile when used with device lambdas.
// UNSUPPORTED: clang && (!nvcc)

// <cuda/std/functional>

// result_of<Fn(ArgTypes...)>

#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class Ret, class Fn>
__host__ __device__ void test_lambda(Fn&&)
{
  static_assert(cuda::std::is_same_v<Ret, typename cuda::std::result_of<Fn()>::type>);

  static_assert(cuda::std::is_same_v<Ret, typename cuda::std::invoke_result<Fn>::type>);
}

int main(int, char**)
{
#if TEST_CUDA_COMPILER(NVCC) || TEST_COMPILER(NVRTC)
  { // extended device lambda
    test_lambda<int>([] __device__() {
      return 42;
    });
    test_lambda<double>([] __device__() {
      return 42.0;
    });
  }
#endif // TEST_CUDA_COMPILER(NVCC) || TEST_COMPILER(NVRTC)

  return 0;
}
