//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <functional>
//
// reference_wrapper
//
// template <class U>
//   reference_wrapper(U&&);

// #include <cuda/std/functional>
#include <cuda/std/cassert>
#include <cuda/std/utility>

#include "test_macros.h"

struct B
{};

struct A1
{
  mutable B b_;
  __host__ __device__ TEST_CONSTEXPR operator B&() const
  {
    return b_;
  }
};

struct A2
{
  mutable B b_;
  __host__ __device__ TEST_CONSTEXPR operator B&() const TEST_NOEXCEPT
  {
    return b_;
  }
};

__host__ __device__ void implicitly_convert(cuda::std::reference_wrapper<B>) TEST_NOEXCEPT;

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  {
    A1 a{};
#ifndef TEST_COMPILER_BROKEN_SMF_NOEXCEPT
    ASSERT_NOT_NOEXCEPT(implicitly_convert(a));
#endif // TEST_COMPILER_BROKEN_SMF_NOEXCEPT
    cuda::std::reference_wrapper<B> b1 = a;
    assert(&b1.get() == &a.b_);
#ifndef TEST_COMPILER_BROKEN_SMF_NOEXCEPT
    ASSERT_NOT_NOEXCEPT(b1 = a);
#endif // TEST_COMPILER_BROKEN_SMF_NOEXCEPT
    b1 = a;
    assert(&b1.get() == &a.b_);
  }
  {
    A2 a{};
    ASSERT_NOEXCEPT(implicitly_convert(a));
    cuda::std::reference_wrapper<B> b2 = a;
    assert(&b2.get() == &a.b_);
    ASSERT_NOEXCEPT(b2 = a);
    b2 = a;
    assert(&b2.get() == &a.b_);
  }
  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER > 2017 && !defined(TEST_COMPILER_NVRTC)
  static_assert(test());
#endif

  return 0;
}
