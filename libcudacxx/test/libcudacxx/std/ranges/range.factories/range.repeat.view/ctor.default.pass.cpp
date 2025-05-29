//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// repeat_view() requires default_initializable<T> = default;

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/ranges>

struct DefaultInt42
{
  int value = 42;
};

struct Int
{
  __host__ __device__ Int(int) {}
};

static_assert(cuda::std::default_initializable<cuda::std::ranges::repeat_view<DefaultInt42>>);
static_assert(!cuda::std::default_initializable<cuda::std::ranges::repeat_view<Int>>);

__host__ __device__ constexpr bool test()
{
  cuda::std::ranges::repeat_view<DefaultInt42> rv;
  assert((*rv.begin()).value == 42);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
