//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// constexpr bool has_value() const noexcept;

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/expected>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

// Test noexcept
template <class T, class = void>
constexpr bool HasValueNoexcept = false;

template <class T>
constexpr bool HasValueNoexcept<T, cuda::std::void_t<decltype(cuda::std::declval<T>().has_value())>> =
  noexcept(cuda::std::declval<T>().has_value());

struct Foo
{};
static_assert(!HasValueNoexcept<Foo>, "");

static_assert(HasValueNoexcept<cuda::std::expected<int, int>>, "");
static_assert(HasValueNoexcept<const cuda::std::expected<int, int>>, "");

__host__ __device__ constexpr bool test()
{
  // has_value
  {
    const cuda::std::expected<int, int> e(5);
    assert(e.has_value());
  }

  // !has_value
  {
    const cuda::std::expected<int, int> e(cuda::std::unexpect, 5);
    assert(!e.has_value());
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
