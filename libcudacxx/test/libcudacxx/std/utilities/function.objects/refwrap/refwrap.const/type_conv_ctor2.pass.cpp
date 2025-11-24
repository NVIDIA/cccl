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
  __host__ __device__ constexpr operator B&() const
  {
    return b_;
  }
};

struct A2
{
  mutable B b_;
  __host__ __device__ constexpr operator B&() const noexcept
  {
    return b_;
  }
};

__host__ __device__ void implicitly_convert(cuda::std::reference_wrapper<B>) noexcept;

__host__ __device__ constexpr bool test()
{
  {
    A1 a{};
#if !_CCCL_COMPILER(GCC, <, 9) && !(_CCCL_COMPILER(MSVC, <, 19, 40) && _CCCL_STD_VER < 2020)
    static_assert(!noexcept(implicitly_convert(a)));
#endif // !_CCCL_COMPILER(GCC, <, 8) && !(_CCCL_COMPILER(MSVC, <, 19, 40) && _CCCL_STD_VER < 2020)
    cuda::std::reference_wrapper<B> b1 = a;
    assert(&b1.get() == &a.b_);
#if !_CCCL_COMPILER(GCC, <, 9) && !(_CCCL_COMPILER(MSVC, <, 19, 40) && _CCCL_STD_VER < 2020)
    static_assert(!noexcept(b1 = a));
#endif // !_CCCL_COMPILER(GCC, <, 8) && !(_CCCL_COMPILER(MSVC, <, 19, 40) && _CCCL_STD_VER < 2020)
    b1 = a;
    assert(&b1.get() == &a.b_);
  }
  {
    A2 a{};
    static_assert(noexcept(implicitly_convert(a)));
    cuda::std::reference_wrapper<B> b2 = a;
    assert(&b2.get() == &a.b_);
    static_assert(noexcept(b2 = a));
    b2 = a;
    assert(&b2.get() == &a.b_);
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
