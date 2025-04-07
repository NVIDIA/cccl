//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr T* end() noexcept;
// constexpr const T* end() const noexcept;

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "test_macros.h"

struct Empty
{};
struct BigType
{
  char buffer[64] = {10};
};

__host__ __device__ constexpr bool test()
{
  {
    auto sv = cuda::std::ranges::single_view<int>(42);
    assert(sv.end() == sv.begin() + 1);

    static_assert(cuda::std::is_same_v<decltype(sv.end()), int*>);
    static_assert(noexcept(sv.end()));
  }
  {
    const auto sv = cuda::std::ranges::single_view<int>(42);
    assert(sv.end() == sv.begin() + 1);

    static_assert(cuda::std::is_same_v<decltype(sv.end()), const int*>);
    static_assert(noexcept(sv.end()));
  }

  {
    auto sv = cuda::std::ranges::single_view<Empty>(Empty());
    assert(sv.end() == sv.begin() + 1);

    static_assert(cuda::std::is_same_v<decltype(sv.end()), Empty*>);
  }
  {
    const auto sv = cuda::std::ranges::single_view<Empty>(Empty());
    assert(sv.end() == sv.begin() + 1);

    static_assert(cuda::std::is_same_v<decltype(sv.end()), const Empty*>);
  }

  {
    auto sv = cuda::std::ranges::single_view<BigType>(BigType());
    assert(sv.end() == sv.begin() + 1);

    static_assert(cuda::std::is_same_v<decltype(sv.end()), BigType*>);
  }
  {
    const auto sv = cuda::std::ranges::single_view<BigType>(BigType());
    assert(sv.end() == sv.begin() + 1);

    static_assert(cuda::std::is_same_v<decltype(sv.end()), const BigType*>);
  }

  return true;
}

int main(int, char**)
{
  test();
#if defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test());
#endif // _CCCL_BUILTIN_ADDRESSOF

  return 0;
}
