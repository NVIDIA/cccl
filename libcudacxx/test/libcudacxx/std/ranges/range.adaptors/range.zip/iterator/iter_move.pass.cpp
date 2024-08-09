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

// friend constexpr auto iter_move(const iterator& i) noexcept(see below);

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/iterator>
#include <cuda/std/ranges>
#include <cuda/std/tuple>

#include "../types.h"
#include "test_macros.h"

struct ThrowingMove
{
  ThrowingMove() = default;
  __host__ __device__ constexpr ThrowingMove(ThrowingMove&&){};
};

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  {
    // underlying iter_move noexcept
    cuda::std::array<int, 4> a1{1, 2, 3, 4};
    const cuda::std::array<double, 2> a2{3.0, 4.0};

    cuda::std::ranges::zip_view v(a1, a2, cuda::std::views::iota(3L));
    assert(cuda::std::ranges::iter_move(v.begin()) == cuda::std::make_tuple(1, 3.0, 3L));
    static_assert(cuda::std::is_same_v<decltype(cuda::std::ranges::iter_move(v.begin())),
                                       cuda::std::tuple<int&&, const double&&, long>>);

    auto it = v.begin();
    static_assert(noexcept(cuda::std::ranges::iter_move(it)));
  }

#if !defined(TEST_COMPILER_NVRTC)
  {
    // underlying iter_move may throw
    auto throwingMoveRange = cuda::std::views::iota(0, 2) | cuda::std::views::transform([](auto) noexcept {
                               return ThrowingMove{};
                             });
    cuda::std::ranges::zip_view v(throwingMoveRange);
    auto it = v.begin();

#  if !defined(TEST_COMPILER_ICC) // broken noexcept
    static_assert(!noexcept(cuda::std::ranges::iter_move(it)));
#  endif // !TEST_COMPILER_ICC
  }
#endif // !TEST_COMPILER_NVRTC

  {
    // underlying iterators' iter_move are called through ranges::iter_move
    adltest::IterMoveSwapRange r1{}, r2{};
    assert(r1.iter_move_called_times == 0);
    assert(r2.iter_move_called_times == 0);
    cuda::std::ranges::zip_view v(r1, r2);
    auto it = v.begin();
    {
      auto&& i = cuda::std::ranges::iter_move(it);
      unused(i);
      assert(r1.iter_move_called_times == 1);
      assert(r2.iter_move_called_times == 1);
    }
    {
      auto&& i = cuda::std::ranges::iter_move(it);
      unused(i);
      assert(r1.iter_move_called_times == 2);
      assert(r2.iter_move_called_times == 2);
    }
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2020 && defined(_LIBCUDACXX_ADDRESSOF)
  static_assert(test(), "");
#endif // TEST_STD_VER > 2017 && defined(_LIBCUDACXX_ADDRESSOF)

  return 0;
}
