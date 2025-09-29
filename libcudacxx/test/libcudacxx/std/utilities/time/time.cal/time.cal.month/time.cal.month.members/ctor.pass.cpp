//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <chrono>
// class month;

//                     month() = default;
//  explicit constexpr month(int m) noexcept;
//  explicit constexpr operator int() const noexcept;

//  Effects: Constructs an object of type month by initializing m_ with m.
//    The value held is unspecified if d is not in the range [0, 255].

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  using month = cuda::std::chrono::month;

  static_assert(noexcept(month{}));
  static_assert(noexcept(month(1)));
  static_assert(noexcept(static_cast<unsigned>(month(1))));

  {
    month m{};
    assert(static_cast<unsigned>(m) == 0);
  }

  {
    month m{1};
    assert(static_cast<unsigned>(m) == 1);
  }

  for (unsigned i = 0; i <= 255; ++i)
  {
    month m(i);
    assert(static_cast<unsigned>(m) == i);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
