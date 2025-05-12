//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr const W & operator*() const noexcept;

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/ranges>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  // unbound
  {
    cuda::std::ranges::repeat_view<int> v(31);
    auto iter = v.begin();

    const int& val = *iter;
    for (int i = 0; i < 100; ++i, ++iter)
    {
      assert(*iter == 31);
      assert(&*iter == &val);
    }

    static_assert(noexcept(*iter));
    static_assert(cuda::std::same_as<decltype(*iter), const int&>);
  }

  // bound && one element
  {
    cuda::std::ranges::repeat_view<int, int> v(31, 1);
    auto iter = v.begin();
    assert(*iter == 31);
    static_assert(noexcept(*iter));
    static_assert(cuda::std::same_as<decltype(*iter), const int&>);
  }

  // bound && several elements
  {
    cuda::std::ranges::repeat_view<int, int> v(31, 100);
    auto iter = v.begin();

    const int& val = *iter;
    for (int i = 0; i < 100; ++i, ++iter)
    {
      assert(*iter == 31);
      assert(&*iter == &val);
    }
  }

#if TEST_STD_VER >= 2023 // requires lifetime expansion of the temporary view
  // bound && foreach
  {
    for (const auto& val : cuda::std::views::repeat(31, 100))
    {
      assert(val == 31);
    }
  }
#endif // TEST_STD_VER >= 2023

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
