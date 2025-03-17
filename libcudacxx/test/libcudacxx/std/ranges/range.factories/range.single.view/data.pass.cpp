//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr T* data() noexcept;
// constexpr const T* data() const noexcept;

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
    assert(*sv.data() == 42);

    static_assert(cuda::std::is_same_v<decltype(sv.data()), int*>);
    static_assert(noexcept(sv.data()));
  }
  {
    const auto sv = cuda::std::ranges::single_view<int>(42);
    assert(*sv.data() == 42);

    static_assert(cuda::std::is_same_v<decltype(sv.data()), const int*>);
    static_assert(noexcept(sv.data()));
  }

  {
    auto sv = cuda::std::ranges::single_view<Empty>(Empty());
    assert(sv.data() != nullptr);

    static_assert(cuda::std::is_same_v<decltype(sv.data()), Empty*>);
  }
  {
    const auto sv = cuda::std::ranges::single_view<Empty>(Empty());
    assert(sv.data() != nullptr);

    static_assert(cuda::std::is_same_v<decltype(sv.data()), const Empty*>);
  }

  {
    auto sv = cuda::std::ranges::single_view<BigType>(BigType());
    assert(sv.data()->buffer[0] == 10);

    static_assert(cuda::std::is_same_v<decltype(sv.data()), BigType*>);
  }
  {
    const auto sv = cuda::std::ranges::single_view<BigType>(BigType());
    assert(sv.data()->buffer[0] == 10);

    static_assert(cuda::std::is_same_v<decltype(sv.data()), const BigType*>);
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
