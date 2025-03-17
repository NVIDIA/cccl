//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <mdspan>

// Test default construction:
//
// constexpr default_accessor() noexcept = default;

#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/mdspan>
#include <cuda/std/type_traits>

#include "../MinimalElementType.h"
#include "test_macros.h"

template <class T>
__host__ __device__ constexpr void test_construction()
{
  static_assert(noexcept(cuda::std::default_accessor<T>{}));
  cuda::std::default_accessor<T> acc{};
  static_assert(cuda::std::is_trivially_default_constructible<cuda::std::default_accessor<T>>::value);
  unused(acc);
}

__host__ __device__ constexpr bool test()
{
  test_construction<int>();
  test_construction<const int>();
  test_construction<MinimalElementType>();
  test_construction<const MinimalElementType>();
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");
  return 0;
}
