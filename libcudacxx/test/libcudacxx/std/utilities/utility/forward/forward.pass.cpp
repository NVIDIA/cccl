//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// test forward

#include <cuda/std/cassert>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

struct A
{};

__host__ __device__ A source() TEST_NOEXCEPT
{
  return A();
}
__host__ __device__ const A csource() TEST_NOEXCEPT
{
  return A();
}

#if TEST_STD_VER > 2011
__host__ __device__ constexpr bool test_constexpr_forward()
{
  int x        = 42;
  const int cx = 101;
  return cuda::std::forward<int&>(x) == 42 && cuda::std::forward<int>(x) == 42
      && cuda::std::forward<const int&>(x) == 42 && cuda::std::forward<const int>(x) == 42
      && cuda::std::forward<int&&>(x) == 42 && cuda::std::forward<const int&&>(x) == 42
      && cuda::std::forward<const int&>(cx) == 101 && cuda::std::forward<const int>(cx) == 101;
}
#endif

int main(int, char**)
{
  A a;
  const A ca = A();

  ((void) a); // Prevent unused warning
  ((void) ca); // Prevent unused warning

  static_assert(cuda::std::is_same<decltype(cuda::std::forward<A&>(a)), A&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward<A>(a)), A&&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward<A>(source())), A&&>::value, "");
  ASSERT_NOEXCEPT(cuda::std::forward<A&>(a));
  ASSERT_NOEXCEPT(cuda::std::forward<A>(a));
  ASSERT_NOEXCEPT(cuda::std::forward<A>(source()));

  static_assert(cuda::std::is_same<decltype(cuda::std::forward<const A&>(a)), const A&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward<const A>(a)), const A&&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward<const A>(source())), const A&&>::value, "");
  ASSERT_NOEXCEPT(cuda::std::forward<const A&>(a));
  ASSERT_NOEXCEPT(cuda::std::forward<const A>(a));
  ASSERT_NOEXCEPT(cuda::std::forward<const A>(source()));

  static_assert(cuda::std::is_same<decltype(cuda::std::forward<const A&>(ca)), const A&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward<const A>(ca)), const A&&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward<const A>(csource())), const A&&>::value, "");
  ASSERT_NOEXCEPT(cuda::std::forward<const A&>(ca));
  ASSERT_NOEXCEPT(cuda::std::forward<const A>(ca));
  ASSERT_NOEXCEPT(cuda::std::forward<const A>(csource()));

#if TEST_STD_VER > 2011
  {
    constexpr int i2 = cuda::std::forward<int>(42);
    static_assert(cuda::std::forward<int>(42) == 42, "");
    static_assert(cuda::std::forward<const int&>(i2) == 42, "");
    static_assert(test_constexpr_forward(), "");
  }
#endif
#if TEST_STD_VER == 2011 && defined(_LIBCUDACXX_VERSION)
  // Test that cuda::std::forward is constexpr in C++11. This is an extension
  // provided by both libc++ and libstdc++.
  {
    constexpr int i2 = cuda::std::forward<int>(42);
    static_assert(cuda::std::forward<int>(42) == 42, "");
    static_assert(cuda::std::forward<const int&>(i2) == 42, "");
  }
#endif

  return 0;
}
