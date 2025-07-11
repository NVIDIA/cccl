//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// static constexpr size_t size() noexcept;

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  {
    auto sv = cuda::std::ranges::single_view<int>(42);
    unused(sv);
    assert(sv.size() == 1);

    static_assert(cuda::std::is_same_v<decltype(sv.size()), size_t>);
    static_assert(noexcept(sv.size()));
  }
  {
    const auto sv = cuda::std::ranges::single_view<int>(42);
    assert(sv.size() == 1);

    static_assert(cuda::std::is_same_v<decltype(sv.size()), size_t>);
    static_assert(noexcept(sv.size()));
  }
  {
    auto sv = cuda::std::ranges::single_view<int>(42);
    assert(cuda::std::ranges::size(sv) == 1);

    static_assert(cuda::std::is_same_v<decltype(cuda::std::ranges::size(sv)), size_t>);
    static_assert(noexcept(cuda::std::ranges::size(sv)));
  }
  {
    const auto sv = cuda::std::ranges::single_view<int>(42);
    assert(cuda::std::ranges::size(sv) == 1);

    static_assert(cuda::std::is_same_v<decltype(cuda::std::ranges::size(sv)), size_t>);
    static_assert(noexcept(cuda::std::ranges::size(sv)));
  }

  // Test that it's static.
  {
    assert(cuda::std::ranges::single_view<int>::size() == 1);

    static_assert(cuda::std::is_same_v<decltype(cuda::std::ranges::single_view<int>::size()), size_t>);
    static_assert(noexcept(cuda::std::ranges::single_view<int>::size()));
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
