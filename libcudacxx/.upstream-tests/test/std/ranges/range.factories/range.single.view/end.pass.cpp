//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// constexpr T* end() noexcept;
// constexpr const T* end() const noexcept;

#include <cuda/std/ranges>
#include <cuda/std/cassert>

#include "test_macros.h"

struct Empty {};
struct BigType { char buffer[64] = {10}; };

__host__ __device__ constexpr bool test() {
  {
    auto sv = cuda::std::ranges::single_view<int>(42);
    assert(sv.end() == sv.begin() + 1);

    ASSERT_SAME_TYPE(decltype(sv.end()), int*);
    static_assert(noexcept(sv.end()));
  }
  {
    const auto sv = cuda::std::ranges::single_view<int>(42);
    assert(sv.end() == sv.begin() + 1);

    ASSERT_SAME_TYPE(decltype(sv.end()), const int*);
    static_assert(noexcept(sv.end()));
  }

  {
    auto sv = cuda::std::ranges::single_view<Empty>(Empty());
    assert(sv.end() == sv.begin() + 1);

    ASSERT_SAME_TYPE(decltype(sv.end()), Empty*);
  }
  {
    const auto sv = cuda::std::ranges::single_view<Empty>(Empty());
    assert(sv.end() == sv.begin() + 1);

    ASSERT_SAME_TYPE(decltype(sv.end()), const Empty*);
  }

  {
    auto sv = cuda::std::ranges::single_view<BigType>(BigType());
    assert(sv.end() == sv.begin() + 1);

    ASSERT_SAME_TYPE(decltype(sv.end()), BigType*);
  }
  {
    const auto sv = cuda::std::ranges::single_view<BigType>(BigType());
    assert(sv.end() == sv.begin() + 1);

    ASSERT_SAME_TYPE(decltype(sv.end()), const BigType*);
  }

  return true;
}

int main(int, char**) {
  test();
#if defined(_LIBCUDACXX_ADDRESSOF) \
 && !defined(TEST_COMPILER_NVCC_BELOW_11_3)
  static_assert(test());
#endif // _LIBCUDACXX_ADDRESSOF && !defined(TEST_COMPILER_NVCC_BELOW_11_3)

  return 0;
}
