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

// constexpr unreachable_sentinel_t end() const noexcept;
// constexpr iterator end() const requires (!same_as<Bound, unreachable_sentinel_t>);

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/iterator>
#include <cuda/std/ranges>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  // bound
  {
    cuda::std::ranges::repeat_view<int, int> rv(0, 10);
    assert(rv.begin() + 10 == rv.end());
    decltype(auto) iter = rv.end();
    static_assert(cuda::std::same_as<cuda::std::ranges::iterator_t<decltype(rv)>, decltype(iter)>);
    static_assert(cuda::std::same_as<decltype(*iter), const int&>);
    for (const auto& i : rv)
    {
      assert(i == 0);
    }
    unused(iter);
  }

  // unbound
  {
    cuda::std::ranges::repeat_view<int> rv(0);
    assert(rv.begin() + 10 != rv.end());
    decltype(auto) iter = rv.end();
    static_assert(cuda::std::same_as<cuda::std::unreachable_sentinel_t, decltype(iter)>);
    static_assert(noexcept(rv.end()));
    for (const auto& i : rv | cuda::std::views::take(10))
    {
      assert(i == 0);
    }
    unused(iter);
  }
  return true;
}

int main(int, char**)
{
  test();
#if !defined(TEST_COMPILER_CUDACC_BELOW_11_3)
  static_assert(test());
#endif // !TEST_COMPILER_CUDACC_BELOW_11_3

  return 0;
}
