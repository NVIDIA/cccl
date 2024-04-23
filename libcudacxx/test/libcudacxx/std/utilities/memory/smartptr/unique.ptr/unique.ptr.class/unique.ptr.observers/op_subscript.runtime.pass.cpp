//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// <memory>

// unique_ptr

// test op[](size_t)

#include <cuda/std/__memory_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

// TODO: Move TEST_IS_CONSTANT_EVALUATED_CXX23() into it's own header
#include "test_macros.h"

#if defined(TEST_COMPILER_NVCC) || defined(TEST_COMPILER_NVRTC)
TEST_NV_DIAG_SUPPRESS(3060) // call to __builtin_is_constant_evaluated appearing in a non-constexpr function
#endif // TEST_COMPILER_NVCC || TEST_COMPILER_NVRTC
#if defined(TEST_COMPILER_GCC)
#  pragma GCC diagnostic ignored "-Wtautological-compare"
#elif defined(TEST_COMPILER_CLANG)
#  pragma clang diagnostic ignored "-Wtautological-compare"
#endif

STATIC_TEST_GLOBAL_VAR int A_next_ = 0;
class A
{
  int state_;

public:
  __host__ __device__ TEST_CONSTEXPR_CXX23 A()
      : state_(0)
  {
    if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
    {
      state_ = ++A_next_;
    }
  }

  __host__ __device__ TEST_CONSTEXPR_CXX23 int get() const
  {
    return state_;
  }

  __host__ __device__ friend TEST_CONSTEXPR_CXX23 bool operator==(const A& x, int y)
  {
    return x.state_ == y;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX23 A& operator=(int i)
  {
    state_ = i;
    return *this;
  }
};

__host__ __device__ TEST_CONSTEXPR_CXX23 bool test()
{
  cuda::std::unique_ptr<A[]> p(new A[3]);
  if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
  {
    assert(p[0] == 1);
    assert(p[1] == 2);
    assert(p[2] == 3);
  }
  p[0] = 3;
  p[1] = 2;
  p[2] = 1;
  assert(p[0] == 3);
  assert(p[1] == 2);
  assert(p[2] == 1);

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2023
  static_assert(test());
#endif // TEST_STD_VER >= 2023

  return 0;
}
