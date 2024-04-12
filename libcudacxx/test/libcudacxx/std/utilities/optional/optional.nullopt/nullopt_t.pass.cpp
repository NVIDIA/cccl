//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// <cuda/std/optional>

// struct nullopt_t{see below};
// inline constexpr nullopt_t nullopt(unspecified);

// [optional.nullopt]/2:
//   Type nullopt_t shall not have a default constructor or an initializer-list
//   constructor, and shall not be an aggregate.

#include <cuda/std/optional>
#include <cuda/std/type_traits>

#include "test_macros.h"

using cuda::std::nullopt;
using cuda::std::nullopt_t;

__host__ __device__ constexpr bool test()
{
  nullopt_t foo{nullopt};
  unused(foo);
  return true;
}

int main(int, char**)
{
  static_assert(cuda::std::is_empty_v<nullopt_t>, "");
  static_assert(!cuda::std::is_default_constructible_v<nullopt_t>, "");

  static_assert(cuda::std::is_same_v<const nullopt_t, decltype(nullopt)>, "");
  static_assert(test(), "");

  return 0;
}
