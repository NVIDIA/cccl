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

// friend constexpr iterator operator+(iterator i, difference_type n);
// friend constexpr iterator operator+(difference_type n, iterator i);

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/ranges>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  cuda::std::ranges::repeat_view<int> v(10);
  using Iter = cuda::std::ranges::iterator_t<cuda::std::ranges::repeat_view<int>>;
  auto iter  = v.begin();
  assert(iter + 5 == v.begin() + 5);
  assert(5 + iter == v.begin() + 5);
  assert(2 + iter == v.begin() + 2);
  assert(3 + iter == v.begin() + 3);

  static_assert(cuda::std::same_as<decltype(iter + 5), Iter>);
  static_assert(cuda::std::same_as<decltype(5 + iter), Iter>);

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
