//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr iterator& operator++();
// constexpr void operator++(int);

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/ranges>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  using Iter = cuda::std::ranges::iterator_t<cuda::std::ranges::repeat_view<int>>;
  cuda::std::ranges::repeat_view<int> rv(10);
  using Iter = cuda::std::ranges::iterator_t<cuda::std::ranges::repeat_view<int>>;
  auto iter  = rv.begin();

  assert(iter++ == rv.begin());
  assert(++iter == rv.begin() + 2);

  static_assert(cuda::std::same_as<decltype(iter++), Iter>);
  static_assert(cuda::std::same_as<decltype(++iter), Iter&>);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
