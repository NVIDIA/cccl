//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr iterator& operator+=(difference_type n);

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/ranges>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  cuda::std::ranges::repeat_view<int> v(10);
  using Iter = cuda::std::ranges::iterator_t<cuda::std::ranges::repeat_view<int>>;
  auto iter1 = v.begin() + 10;
  auto iter2 = v.begin() + 10;
  assert(iter1 == iter2);
  iter1 += 5;
  assert(iter1 != iter2);
  assert(iter1 == iter2 + 5);

  static_assert(cuda::std::same_as<decltype(iter2 += 5), Iter&>);
  assert(cuda::std::addressof(iter2) == cuda::std::addressof(iter2 += 5));

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
