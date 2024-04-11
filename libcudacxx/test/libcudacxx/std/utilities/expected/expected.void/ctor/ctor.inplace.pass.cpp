//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// constexpr explicit expected(in_place_t) noexcept;

#include <cuda/std/cassert>
#include <cuda/std/expected>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

// test explicit
static_assert(cuda::std::is_constructible_v<cuda::std::expected<void, int>, cuda::std::in_place_t>, "");
static_assert(!cuda::std::is_convertible_v<cuda::std::in_place_t, cuda::std::expected<void, int>>, "");

// test noexcept
static_assert(cuda::std::is_nothrow_constructible_v<cuda::std::expected<void, int>, cuda::std::in_place_t>, "");

__host__ __device__ constexpr bool test()
{
  cuda::std::expected<void, int> e(cuda::std::in_place);
  assert(e.has_value());

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");
  return 0;
}
