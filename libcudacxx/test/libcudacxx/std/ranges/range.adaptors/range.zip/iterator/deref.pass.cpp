//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// constexpr auto operator*() const;

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/tuple>

#include "../types.h"

__host__ __device__ constexpr bool test()
{
  cuda::std::array<int, 4> a{1, 2, 3, 4};
  cuda::std::array<double, 3> b{4.1, 3.2, 4.3};
  {
    // single range
    cuda::std::ranges::zip_view v(a);
    auto it = v.begin();
    assert(&(cuda::std::get<0>(*it)) == &(a[0]));
    static_assert(cuda::std::is_same_v<decltype(*it), cuda::std::tuple<int&>>);
  }

  {
    // operator* is const
    cuda::std::ranges::zip_view v(a);
    const auto it = v.begin();
    assert(&(cuda::std::get<0>(*it)) == &(a[0]));
  }

  {
    // two ranges with different types
    cuda::std::ranges::zip_view v(a, b);
    auto it     = v.begin();
    auto [x, y] = *it;
    assert(&x == &(a[0]));
    assert(&y == &(b[0]));
    static_assert(cuda::std::is_same_v<decltype(*it), cuda::std::pair<int&, double&>>);

    x = 5;
    y = 0.1;
    assert(a[0] == 5);
    assert(b[0] == 0.1);
  }

  {
    // underlying range with prvalue range_reference_t
    cuda::std::ranges::zip_view v(a, b, cuda::std::views::iota(0, 5));
    auto it = v.begin();
    assert(&(cuda::std::get<0>(*it)) == &(a[0]));
    assert(&(cuda::std::get<1>(*it)) == &(b[0]));
    assert(cuda::std::get<2>(*it) == 0);
    static_assert(cuda::std::is_same_v<decltype(*it), cuda::std::tuple<int&, double&, int>>);
  }

  {
    // const-correctness
    cuda::std::ranges::zip_view v(a, cuda::std::as_const(a));
    auto it = v.begin();
    assert(&(cuda::std::get<0>(*it)) == &(a[0]));
    assert(&(cuda::std::get<1>(*it)) == &(a[0]));
    static_assert(cuda::std::is_same_v<decltype(*it), cuda::std::pair<int&, int const&>>);
  }
  return true;
}

int main(int, char**)
{
  test();
#if defined(_LIBCUDACXX_ADDRESSOF)
  static_assert(test(), "");
#endif // defined(_LIBCUDACXX_ADDRESSOF)

  return 0;
}
