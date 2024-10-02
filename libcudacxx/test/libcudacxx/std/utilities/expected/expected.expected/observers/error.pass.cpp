//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// constexpr const E& error() const & noexcept;
// constexpr E& error() & noexcept;
// constexpr E&& error() && noexcept;
// constexpr const E&& error() const && noexcept;

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/expected>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

// Test noexcept
template <class T, class = void>
constexpr bool ErrorNoexcept = false;

template <class T>
constexpr bool ErrorNoexcept<T, cuda::std::void_t<decltype(cuda::std::declval<T>().error())>> =
  noexcept(cuda::std::declval<T>().error());

static_assert(!ErrorNoexcept<int>, "");

static_assert(ErrorNoexcept<cuda::std::expected<int, int>&>, "");
static_assert(ErrorNoexcept<const cuda::std::expected<int, int>&>, "");
static_assert(ErrorNoexcept<cuda::std::expected<int, int>&&>, "");
static_assert(ErrorNoexcept<const cuda::std::expected<int, int>&&>, "");

__host__ __device__ constexpr bool test()
{
  // non-const &
  {
    cuda::std::expected<int, int> e(cuda::std::unexpect, 5);
    decltype(auto) x = e.error();
    static_assert(cuda::std::same_as<decltype(x), int&>, "");
    assert(x == 5);
  }

  // const &
  {
    const cuda::std::expected<int, int> e(cuda::std::unexpect, 5);
    decltype(auto) x = e.error();
    static_assert(cuda::std::same_as<decltype(x), const int&>, "");
    assert(x == 5);
  }

  // non-const &&
  {
    cuda::std::expected<int, int> e(cuda::std::unexpect, 5);
    decltype(auto) x = cuda::std::move(e).error();
    static_assert(cuda::std::same_as<decltype(x), int&&>, "");
    assert(x == 5);
  }

  // const &&
  {
    const cuda::std::expected<int, int> e(cuda::std::unexpect, 5);
    decltype(auto) x = cuda::std::move(e).error();
    static_assert(cuda::std::same_as<decltype(x), const int&&>, "");
    assert(x == 5);
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
