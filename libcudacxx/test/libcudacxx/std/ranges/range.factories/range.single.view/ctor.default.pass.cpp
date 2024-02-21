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

// single_view() requires default_initializable<T> = default;

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "test_macros.h"

struct BigType
{
  char buffer[64] = {10};
};

#if TEST_STD_VER >= 2020
template <bool DefaultCtorEnabled>
struct IsDefaultConstructible
{
  IsDefaultConstructible()
    requires DefaultCtorEnabled
  = default;
};
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <bool DefaultCtorEnabled>
struct IsDefaultConstructible
{};

template <>
struct IsDefaultConstructible<false>
{
  IsDefaultConstructible() = delete;
};
#endif // TEST_STD_VER <= 2017

__host__ __device__ constexpr bool test()
{
  static_assert(cuda::std::default_initializable<cuda::std::ranges::single_view<IsDefaultConstructible<true>>>);
  static_assert(!cuda::std::default_initializable<cuda::std::ranges::single_view<IsDefaultConstructible<false>>>);

  {
    cuda::std::ranges::single_view<BigType> sv;
    assert(sv.data()->buffer[0] == 10);
    assert(sv.size() == 1);
  }
  {
    const cuda::std::ranges::single_view<BigType> sv;
    assert(sv.data()->buffer[0] == 10);
    assert(sv.size() == 1);
  }

  return true;
}

int main(int, char**)
{
  test();
#if defined(_LIBCUDACXX_ADDRESSOF)
  static_assert(test());
#endif // _LIBCUDACXX_ADDRESSOF

  return 0;
}
