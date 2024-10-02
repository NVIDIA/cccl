//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// <cuda/std/optional>

// constexpr T* optional<T>::operator->();

#include <cuda/std/cassert>
#include <cuda/std/optional>
#include <cuda/std/type_traits>

#include "test_macros.h"

using cuda::std::optional;

struct X
{
  __host__ __device__ int test() noexcept
  {
    return 3;
  }
};

struct Y
{
  __host__ __device__ constexpr int test()
  {
    return 3;
  }
};

__host__ __device__ constexpr int test()
{
  optional<Y> opt{Y{}};
  return opt->test();
}

int main(int, char**)
{
  {
    cuda::std::optional<X> opt;
    unused(opt);
    ASSERT_SAME_TYPE(decltype(opt.operator->()), X*);
    // ASSERT_NOT_NOEXCEPT(opt.operator->());
    // FIXME: This assertion fails with GCC because it can see that
    // (A) operator->() is constexpr, and
    // (B) there is no path through the function that throws.
    // It's arguable if this is the correct behavior for the noexcept
    // operator.
    // Regardless this function should still be noexcept(false) because
    // it has a narrow contract.
  }
  {
    optional<X> opt(X{});
    assert(opt->test() == 3);
  }
  {
#if defined(_CCCL_BUILTIN_ADDRESSOF)
#  if !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
    static_assert(test() == 3, "");
#  endif // !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
#endif
  }

  return 0;
}
