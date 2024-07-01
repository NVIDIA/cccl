//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// sentinel() = default;

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "../types.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  using Sent = cuda::std::ranges::sentinel_t<
    cuda::std::ranges::iota_view<Int42<DefaultTo42>, IntComparableWith<Int42<DefaultTo42>>>>;
  using Iter = cuda::std::ranges::iterator_t<
    cuda::std::ranges::iota_view<Int42<DefaultTo42>, IntComparableWith<Int42<DefaultTo42>>>>;
  assert(Sent() == Iter());

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
