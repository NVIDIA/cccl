//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <chrono>
// class day;

//  constexpr day& operator--() noexcept;
//  constexpr day operator--(int) noexcept;

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

using day = cuda::std::chrono::day;

__host__ __device__ constexpr bool test()
{
  static_assert(noexcept(--(cuda::std::declval<day&>())));
  static_assert(noexcept((cuda::std::declval<day&>())--));

  static_assert(cuda::std::is_same_v<day&, decltype(--cuda::std::declval<day&>())>);
  static_assert(cuda::std::is_same_v<day, decltype(cuda::std::declval<day&>()--)>);

  for (unsigned i = 10; i <= 20; ++i)
  {
    day d(i);
    assert(static_cast<unsigned>(--d) == i - 1);
    assert(static_cast<unsigned>(d--) == i - 1);
    assert(static_cast<unsigned>(d) == i - 2);
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
