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

// constexpr iterator& operator--();
// constexpr iterator operator--(int);

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  using Iter = cuda::std::ranges::iterator_t<cuda::std::ranges::repeat_view<int>>;
  cuda::std::ranges::repeat_view<int> rv(10);
  auto iter = rv.begin() + 10;

  assert(iter-- == rv.begin() + 10);
  assert(--iter == rv.begin() + 8);

  static_assert(cuda::std::same_as<decltype(iter--), Iter>);
  static_assert(cuda::std::same_as<decltype(--iter), Iter&>);

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
