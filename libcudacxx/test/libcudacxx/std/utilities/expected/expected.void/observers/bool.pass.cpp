//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// constexpr explicit operator bool() const noexcept;

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/expected>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

// Test noexcept
template <class T, class = void>
constexpr bool OpBoolNoexcept = false;

template <class T>
constexpr bool OpBoolNoexcept<T, cuda::std::void_t<decltype(static_cast<bool>(cuda::std::declval<T>()))>> =
  noexcept(static_cast<bool>(cuda::std::declval<T>()));

struct Foo
{};
static_assert(!OpBoolNoexcept<Foo>, "");

static_assert(OpBoolNoexcept<cuda::std::expected<void, int>>, "");
static_assert(OpBoolNoexcept<const cuda::std::expected<void, int>>, "");

// Test explicit
static_assert(!cuda::std::is_convertible_v<cuda::std::expected<void, int>, bool>, "");

__host__ __device__ constexpr bool test()
{
  // has_value
  {
    const cuda::std::expected<void, int> e;
    assert(static_cast<bool>(e));
  }

  // !has_value
  {
    const cuda::std::expected<void, int> e(cuda::std::unexpect, 5);
    assert(!static_cast<bool>(e));
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
