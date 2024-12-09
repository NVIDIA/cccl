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

// friend constexpr void iter_swap(const iterator& l, const iterator& r) noexcept(see below)
//   requires (indirectly_swappable<iterator_t<maybe-const<Const, Views>>> && ...);

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "../types.h"
#include "test_macros.h"

struct ThrowingMove
{
  ThrowingMove() = default;
  __host__ __device__ ThrowingMove(ThrowingMove&&) {};
  __host__ __device__ ThrowingMove& operator=(ThrowingMove&&)
  {
    return *this;
  }
};

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  {
    cuda::std::array<int, 4> a1{1, 2, 3, 4};
    cuda::std::array<double, 4> a2{0.1, 0.2, 0.3};
    cuda::std::ranges::zip_view v(a1, a2);
    auto iter1 = v.begin();
    auto iter2 = ++v.begin();

    cuda::std::ranges::iter_swap(iter1, iter2);

    assert(a1[0] == 2);
    assert(a1[1] == 1);
    assert(a2[0] == 0.2);
    assert(a2[1] == 0.1);

    auto [x1, y1] = *iter1;
    assert(&x1 == &a1[0]);
    assert(&y1 == &a2[0]);

    auto [x2, y2] = *iter2;
    assert(&x2 == &a1[1]);
    assert(&y2 == &a2[1]);

#if !defined(TEST_COMPILER_GCC)
    static_assert(noexcept(cuda::std::ranges::iter_swap(iter1, iter2)));
#endif // !TEST_COMPILER_GCC
  }

  {
    // underlying iter_swap may throw
    cuda::std::array<ThrowingMove, 2> iterSwapMayThrow{};
    cuda::std::ranges::zip_view v(iterSwapMayThrow);
    auto iter1 = v.begin();
    auto iter2 = ++v.begin();
#if !defined(TEST_COMPILER_ICC) // broken noexcept
    static_assert(!noexcept(cuda::std::ranges::iter_swap(iter1, iter2)));
#endif // !TEST_COMPILER_ICC
  }

  {
    // underlying iterators' iter_move are called through ranges::iter_swap
    adltest::IterMoveSwapRange r1, r2;
    assert(r1.iter_swap_called_times == 0);
    assert(r2.iter_swap_called_times == 0);

    cuda::std::ranges::zip_view v{r1, r2};
    auto it1 = v.begin();
    auto it2 = cuda::std::ranges::next(it1, 3);

    cuda::std::ranges::iter_swap(it1, it2);
    assert(r1.iter_swap_called_times == 2);
    assert(r2.iter_swap_called_times == 2);

    cuda::std::ranges::iter_swap(it1, it2);
    assert(r1.iter_swap_called_times == 4);
    assert(r2.iter_swap_called_times == 4);
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
