//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <chrono>
// class year;

// constexpr bool is_leap() const noexcept;
//  y_ % 4 == 0 && (y_ % 100 != 0 || y_ % 400 == 0)
//

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  using year = cuda::std::chrono::year;

  static_assert(noexcept(year(1).is_leap()));
  static_assert(cuda::std::is_same_v<bool, decltype(year(1).is_leap())>);

  assert(!year{1}.is_leap());
  assert(!year{2}.is_leap());
  assert(!year{3}.is_leap());
  assert(year{4}.is_leap());

  assert(year{-2000}.is_leap());
  assert(year{-400}.is_leap());
  assert(!year{-300}.is_leap());
  assert(!year{-200}.is_leap());

  assert(!year{200}.is_leap());
  assert(!year{300}.is_leap());
  assert(year{400}.is_leap());
  assert(!year{1997}.is_leap());
  assert(!year{1998}.is_leap());
  assert(!year{1999}.is_leap());
  assert(year{2000}.is_leap());
  assert(!year{2001}.is_leap());
  assert(!year{2002}.is_leap());
  assert(!year{2003}.is_leap());
  assert(year{2004}.is_leap());
  assert(!year{2100}.is_leap());

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
