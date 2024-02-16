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

#include <cuda/std/type_traits>
#include <cuda/std/cassert>

#include "test_macros.h"

#ifndef _CCCL_CUDA_COMPILER
static_assert(false, "There are no device lambdas in host only tests");
#endif // !_CCCL_CUDA_COMPILER

template <class Ret, class Fn>
TEST_HOST_DEVICE
void test_lambda(Fn &&)
{
    ASSERT_SAME_TYPE(Ret, typename cuda::std::result_of<Fn()>::type);

#if TEST_STD_VER > 2011
    ASSERT_SAME_TYPE(Ret, typename cuda::std::invoke_result<Fn>::type);
#endif
}

int main(int, char**)
{
#if defined(TEST_COMPILER_NVCC) || defined (TEST_COMPILER_NVRTC)
    { // extended device lambda
    test_lambda<int>([] TEST_DEVICE () -> int { return 42; });
    test_lambda<double>([] TEST_DEVICE () -> double { return 42.0; });
    }
#endif

  return 0;
}
