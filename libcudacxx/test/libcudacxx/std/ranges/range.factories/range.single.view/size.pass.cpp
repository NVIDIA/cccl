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

    ASSERT_SAME_TYPE(decltype(sv.size()), size_t);
    static_assert(noexcept(sv.size()));
  }
  {
    const auto sv = cuda::std::ranges::single_view<int>(42);
    assert(sv.size() == 1);

    ASSERT_SAME_TYPE(decltype(sv.size()), size_t);
    static_assert(noexcept(sv.size()));
  }
  {
    auto sv = cuda::std::ranges::single_view<int>(42);
    assert(cuda::std::ranges::size(sv) == 1);

    ASSERT_SAME_TYPE(decltype(cuda::std::ranges::size(sv)), size_t);
    static_assert(noexcept(cuda::std::ranges::size(sv)));
  }
  {
    const auto sv = cuda::std::ranges::single_view<int>(42);
    assert(cuda::std::ranges::size(sv) == 1);

    ASSERT_SAME_TYPE(decltype(cuda::std::ranges::size(sv)), size_t);
    static_assert(noexcept(cuda::std::ranges::size(sv)));
  }

  // Test that it's static.
  {
    assert(cuda::std::ranges::single_view<int>::size() == 1);

    ASSERT_SAME_TYPE(decltype(cuda::std::ranges::single_view<int>::size()), size_t);
    static_assert(noexcept(cuda::std::ranges::single_view<int>::size()));
  }

  return true;
}

int main(int, char**)
{
  test();
#if !defined(TEST_COMPILER_CUDACC_BELOW_11_3)
  static_assert(test());
#endif // !defined(TEST_COMPILER_CUDACC_BELOW_11_3)

  return 0;
}
