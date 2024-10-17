//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// template<class T2, class E2> requires (!is_void_v<T2>)
//   friend constexpr bool operator==(const expected& x, const expected<T2, E2>& y);

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/expected>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

// Test constraint
template <class T1, class T2, class = void>
constexpr bool CanCompare = false;

template <class T1, class T2>
constexpr bool CanCompare<T1, T2, cuda::std::void_t<decltype(cuda::std::declval<T1>() == cuda::std::declval<T2>())>> =
  true;

struct Foo
{};
static_assert(!CanCompare<Foo, Foo>, "");

static_assert(CanCompare<cuda::std::expected<int, int>, cuda::std::expected<int, int>>, "");
static_assert(CanCompare<cuda::std::expected<int, int>, cuda::std::expected<short, short>>, "");

// Note this is true because other overloads are unconstrained
static_assert(CanCompare<cuda::std::expected<int, int>, cuda::std::expected<void, int>>, "");

__host__ __device__ constexpr bool test()
{
  // x.has_value() && y.has_value()
  {
    const cuda::std::expected<int, int> e1(5);
    const cuda::std::expected<int, int> e2(10);
    const cuda::std::expected<int, int> e3(5);
    assert(e1 != e2);
    assert(e1 == e3);
  }

  // !x.has_value() && y.has_value()
  {
    const cuda::std::expected<int, int> e1(cuda::std::unexpect, 5);
    const cuda::std::expected<int, int> e2(10);
    const cuda::std::expected<int, int> e3(5);
    assert(e1 != e2);
    assert(e1 != e3);
  }

  // x.has_value() && !y.has_value()
  {
    const cuda::std::expected<int, int> e1(5);
    const cuda::std::expected<int, int> e2(cuda::std::unexpect, 10);
    const cuda::std::expected<int, int> e3(cuda::std::unexpect, 5);
    assert(e1 != e2);
    assert(e1 != e3);
  }

  // !x.has_value() && !y.has_value()
  {
    const cuda::std::expected<int, int> e1(cuda::std::unexpect, 5);
    const cuda::std::expected<int, int> e2(cuda::std::unexpect, 10);
    const cuda::std::expected<int, int> e3(cuda::std::unexpect, 5);
    assert(e1 != e2);
    assert(e1 == e3);
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test(), "");
#endif // TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
  return 0;
}
