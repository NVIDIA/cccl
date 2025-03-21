//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// test forward

#include <cuda/std/cassert>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

struct A
{};

__host__ __device__ A source() noexcept
{
  return A();
}
__host__ __device__ const A csource() noexcept
{
  return A();
}

__host__ __device__ constexpr bool test_constexpr_forward()
{
  int x        = 42;
  const int cx = 101;
  return cuda::std::forward<int&>(x) == 42 && cuda::std::forward<int>(x) == 42
      && cuda::std::forward<const int&>(x) == 42 && cuda::std::forward<const int>(x) == 42
      && cuda::std::forward<int&&>(x) == 42 && cuda::std::forward<const int&&>(x) == 42
      && cuda::std::forward<const int&>(cx) == 101 && cuda::std::forward<const int>(cx) == 101;
}

int main(int, char**)
{
  [[maybe_unused]] A a;
  [[maybe_unused]] const A ca = A();

  static_assert(cuda::std::is_same<decltype(cuda::std::forward<A&>(a)), A&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward<A>(a)), A&&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward<A>(source())), A&&>::value, "");
  static_assert(noexcept(cuda::std::forward<A&>(a)));
  static_assert(noexcept(cuda::std::forward<A>(a)));
  static_assert(noexcept(cuda::std::forward<A>(source())));

  static_assert(cuda::std::is_same<decltype(cuda::std::forward<const A&>(a)), const A&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward<const A>(a)), const A&&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward<const A>(source())), const A&&>::value, "");
  static_assert(noexcept(cuda::std::forward<const A&>(a)));
  static_assert(noexcept(cuda::std::forward<const A>(a)));
  static_assert(noexcept(cuda::std::forward<const A>(source())));

  static_assert(cuda::std::is_same<decltype(cuda::std::forward<const A&>(ca)), const A&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward<const A>(ca)), const A&&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward<const A>(csource())), const A&&>::value, "");
  static_assert(noexcept(cuda::std::forward<const A&>(ca)));
  static_assert(noexcept(cuda::std::forward<const A>(ca)));
  static_assert(noexcept(cuda::std::forward<const A>(csource())));

  {
    constexpr int i2 = cuda::std::forward<int>(42);
    static_assert(cuda::std::forward<int>(42) == 42, "");
    static_assert(cuda::std::forward<const int&>(i2) == 42, "");
    static_assert(test_constexpr_forward(), "");
  }

  return 0;
}
