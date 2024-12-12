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

// constexpr auto size() const requires (!same_as<Bound, unreachable_sentinel_t>);

#include <cuda/std/cassert>
#include <cuda/std/iterator>
#include <cuda/std/limits>
#include <cuda/std/ranges>

#include "test_macros.h"

#if TEST_STD_VER >= 2020
template <class T>
concept has_size = requires(T&& view) {
  { cuda::std::forward<T>(view).size() };
};
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class T, class = void>
_CCCL_INLINE_VAR constexpr bool has_size = false;

template <class T>
_CCCL_INLINE_VAR constexpr bool has_size<T, cuda::std::void_t<decltype(cuda::std::declval<T>().size())>> = true;
#endif // TEST_STD_VER <= 2017

static_assert(has_size<cuda::std::ranges::repeat_view<int, int>>);
static_assert(!has_size<cuda::std::ranges::repeat_view<int>>);
static_assert(!has_size<cuda::std::ranges::repeat_view<int, cuda::std::unreachable_sentinel_t>>);

__host__ __device__ constexpr bool test()
{
  {
    cuda::std::ranges::repeat_view<int, int> rv(10, 20);
    assert(rv.size() == 20);
  }

  {
    constexpr int int_max = cuda::std::numeric_limits<int>::max();
    cuda::std::ranges::repeat_view<int, int> rv(10, int_max);
    assert(rv.size() == int_max);
  }

  return true;
}

int main(int, char**)
{
  test();
#if !defined(TEST_COMPILER_CLANG) || __clang__ > 9
  static_assert(test());
#endif // clang > 9

  return 0;
}
