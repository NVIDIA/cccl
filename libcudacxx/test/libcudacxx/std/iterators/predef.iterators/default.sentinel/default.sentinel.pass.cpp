//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// struct default_sentinel_t;
// inline constexpr default_sentinel_t default_sentinel;

#include <cuda/std/concepts>
#include <cuda/std/iterator>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  static_assert(cuda::std::is_empty_v<cuda::std::default_sentinel_t>);
  static_assert(cuda::std::semiregular<cuda::std::default_sentinel_t>);

  static_assert(cuda::std::same_as<decltype(cuda::std::default_sentinel), const cuda::std::default_sentinel_t>);

  cuda::std::default_sentinel_t s1;
  auto s2 = cuda::std::default_sentinel_t{};
  s2      = s1;
  unused(s2);

  return 0;
}
