//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr iterator begin() const;

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/ranges>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  // Test unbound && non-const view
  {
    cuda::std::ranges::repeat_view<int> rv(0);
    decltype(auto) iter = rv.begin();
    static_assert(cuda::std::same_as<cuda::std::ranges::iterator_t<decltype(rv)>, decltype(iter)>);
    assert(*iter == 0);
  }

  // Test unbound && const view
  {
    const cuda::std::ranges::repeat_view<int> rv(0);
    decltype(auto) iter = rv.begin();
    static_assert(cuda::std::same_as<cuda::std::ranges::iterator_t<decltype(rv)>, decltype(iter)>);
    assert(*iter == 0);
  }

  // Test bound && non-const view
  {
    cuda::std::ranges::repeat_view<int, int> rv(1024, 10);
    decltype(auto) iter = rv.begin();
    static_assert(cuda::std::same_as<cuda::std::ranges::iterator_t<decltype(rv)>, decltype(iter)>);
    assert(*iter == 1024);
  }

  // Test bound && const view
  {
    const cuda::std::ranges::repeat_view<int, long long> rv(1024, 10);
    decltype(auto) iter = rv.begin();
    static_assert(cuda::std::same_as<cuda::std::ranges::iterator_t<decltype(rv)>, decltype(iter)>);
    assert(*iter == 1024);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
