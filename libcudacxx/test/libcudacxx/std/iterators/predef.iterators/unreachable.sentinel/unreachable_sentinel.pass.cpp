//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// struct unreachable_sentinel_t;
// inline constexpr unreachable_sentinel_t unreachable_sentinel;

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/iterator>
#include <cuda/std/type_traits>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  static_assert(cuda::std::is_empty_v<cuda::std::unreachable_sentinel_t>);
  static_assert(cuda::std::semiregular<cuda::std::unreachable_sentinel_t>);

  static_assert(cuda::std::same_as<decltype(cuda::std::unreachable_sentinel), const cuda::std::unreachable_sentinel_t>);

  auto sentinel = cuda::std::unreachable_sentinel;
  int i         = 42;
  assert(i != sentinel);
  assert(sentinel != i);
  assert(!(i == sentinel));
  assert(!(sentinel == i));

  assert(&i != sentinel);
  assert(sentinel != &i);
  assert(!(&i == sentinel));
  assert(!(sentinel == &i));

  int* p = nullptr;
  assert(p != sentinel);
  assert(sentinel != p);
  assert(!(p == sentinel));
  assert(!(sentinel == p));

  static_assert(cuda::std::__weakly_equality_comparable_with<cuda::std::unreachable_sentinel_t, int>);
  static_assert(cuda::std::__weakly_equality_comparable_with<cuda::std::unreachable_sentinel_t, int*>);
#if !defined(TEST_COMPILER_GCC) || __GNUC__ > 11 || TEST_STD_VER < 2020 // gcc 10 has an issue with void
  static_assert(!cuda::std::__weakly_equality_comparable_with<cuda::std::unreachable_sentinel_t, void*>);
#endif // !defined(TEST_COMPILER_GCC) || __GNUC__ > 11 || TEST_STD_VER < 2020
  ASSERT_NOEXCEPT(sentinel == p);
  ASSERT_NOEXCEPT(sentinel != p);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
