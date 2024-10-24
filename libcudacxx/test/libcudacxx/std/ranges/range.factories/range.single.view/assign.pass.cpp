//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// Tests that <value_> is a <copyable-box>.

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/utility>

#include "test_macros.h"

struct Copyable
{
  __host__ __device__ constexpr Copyable() noexcept
      : val_(0)
  {}
  __host__ __device__ constexpr Copyable(const Copyable&)
      : val_(0)
  {}
  __host__ __device__ constexpr Copyable(Copyable&&)
      : val_(0)
  {}
  __host__ __device__ constexpr Copyable& operator=(const Copyable&)
  {
    val_ = 42;
    return *this;
  }

  __host__ __device__ constexpr Copyable& operator=(Copyable&&)
  {
    val_ = 1337;
    return *this;
  }

  int val_ = 0;
};
static_assert(cuda::std::copyable<Copyable>);

struct NotAssignable
{
  NotAssignable()                     = default;
  NotAssignable(const NotAssignable&) = default;
  NotAssignable(NotAssignable&&)      = default;

  NotAssignable& operator=(const NotAssignable&) = delete;
  NotAssignable& operator=(NotAssignable&&)      = delete;
};

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  {
    const cuda::std::ranges::single_view<NotAssignable> a;
    cuda::std::ranges::single_view<NotAssignable> b;
    b = a;
    b = cuda::std::move(a);
    unused(b);
  }

  {
    cuda::std::ranges::single_view<Copyable> a;
    cuda::std::ranges::single_view<Copyable> b;
    b = a;
    assert(b.begin()->val_ == 42);
    b = cuda::std::move(a);
    assert(b.begin()->val_ == 1337);
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2020 && defined(_LIBCUDACXX_ADDRESSOF)
  static_assert(test());
#endif // TEST_STD_VER >= 2020 && _LIBCUDACXX_ADDRESSOF

  return 0;
}
