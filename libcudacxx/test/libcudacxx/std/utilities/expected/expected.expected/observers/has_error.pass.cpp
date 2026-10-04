//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr bool has_error() const noexcept;

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/expected>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

// Test noexcept
template <class T, class = void>
constexpr bool HasErrorNoexcept = false;

template <class T>
constexpr bool HasErrorNoexcept<T, cuda::std::void_t<decltype(cuda::std::declval<T>().has_error())>> =
  noexcept(cuda::std::declval<T>().has_error());

struct Foo
{};
static_assert(!HasErrorNoexcept<Foo>);

static_assert(HasErrorNoexcept<cuda::std::expected<int, int>>);
static_assert(HasErrorNoexcept<const cuda::std::expected<int, int>>);

TEST_FUNC constexpr bool test()
{
  // has_error
  {
    const cuda::std::expected<int, int> e(cuda::std::unexpect, 5);
    assert(e.has_error());
  }

  // !has_error
  {
    const cuda::std::expected<int, int> e(5);
    assert(!e.has_error());
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test());
#endif // TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
  return 0;
}
