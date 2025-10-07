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

// constexpr bool ok() const noexcept;
//  Returns: 1 <= d_ && d_ <= 31

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  using day = cuda::std::chrono::day;

  static_assert(noexcept((cuda::std::declval<const day>().ok())));
  static_assert(cuda::std::is_same_v<bool, decltype(cuda::std::declval<const day>().ok())>);

  assert(!day{0}.ok());
  for (unsigned i = 1; i <= 31; ++i)
  {
    assert(day{i}.ok());
  }
  for (unsigned i = 32; i <= 255; ++i)
  {
    assert(!day{i}.ok());
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
