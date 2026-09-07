//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template<class G = E> constexpr E error_or(G&& e) const &;
// template<class G = E> constexpr E error_or(G&& e) &&;

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/expected>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

struct ConstructFromInt
{
  int value;
  TEST_FUNC constexpr ConstructFromInt(int v)
      : value(v)
  {}
};

TEST_FUNC constexpr bool test_default_template_arg()
{
  // const &, has_value()
  {
    const cuda::std::expected<void, ConstructFromInt> e;
    decltype(auto) x = e.error_or(10);
    static_assert(cuda::std::same_as<ConstructFromInt, decltype(x)>);
    assert(x.value == 10);
  }

  // const &, !has_value()
  {
    const cuda::std::expected<void, ConstructFromInt> e(cuda::std::unexpect, 5);
    decltype(auto) x = e.error_or(10);
    static_assert(cuda::std::same_as<ConstructFromInt, decltype(x)>);
    assert(x.value == 5);
  }

  // &&, has_value()
  {
    cuda::std::expected<void, ConstructFromInt> e;
    decltype(auto) x = cuda::std::move(e).error_or(10);
    static_assert(cuda::std::same_as<ConstructFromInt, decltype(x)>);
    assert(x.value == 10);
  }

  // &&, !has_value()
  {
    cuda::std::expected<void, ConstructFromInt> e(cuda::std::unexpect, 5);
    decltype(auto) x = cuda::std::move(e).error_or(10);
    static_assert(cuda::std::same_as<ConstructFromInt, decltype(x)>);
    assert(x.value == 5);
  }

  return true;
}

TEST_FUNC constexpr bool test()
{
  // const &, has_value()
  {
    const cuda::std::expected<void, int> e;
    decltype(auto) x = e.error_or(10);
    static_assert(cuda::std::same_as<int, decltype(x)>);
    assert(x == 10);
  }

  // const &, !has_value()
  {
    const cuda::std::expected<void, int> e(cuda::std::unexpect, 5);
    decltype(auto) x = e.error_or(10);
    static_assert(cuda::std::same_as<int, decltype(x)>);
    assert(x == 5);
  }

  // &&, has_value()
  {
    cuda::std::expected<void, int> e;
    decltype(auto) x = cuda::std::move(e).error_or(10);
    static_assert(cuda::std::same_as<int, decltype(x)>);
    assert(x == 10);
  }

  // &&, !has_value()
  {
    cuda::std::expected<void, int> e(cuda::std::unexpect, 5);
    decltype(auto) x = cuda::std::move(e).error_or(10);
    static_assert(cuda::std::same_as<int, decltype(x)>);
    assert(x == 5);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  test_default_template_arg();
  static_assert(test_default_template_arg());

  return 0;
}
