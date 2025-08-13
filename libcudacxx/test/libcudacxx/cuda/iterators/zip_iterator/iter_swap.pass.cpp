//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// friend constexpr void iter_swap(const iterator& l, const iterator& r) noexcept(see below)
//   requires (indirectly_swappable<iterator_t<maybe-const<Const, Views>>> && ...);

#include <cuda/iterator>
#include <cuda/std/cassert>

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

struct ThrowingMove
{
  __host__ __device__ constexpr ThrowingMove() noexcept {}
  __host__ __device__ constexpr ThrowingMove(ThrowingMove&&) noexcept(false) {}
  __host__ __device__ ThrowingMove& operator=(ThrowingMove&&) noexcept(false)
  {
    return *this;
  }
};

struct ToThrowingMove
{
  __host__ __device__ constexpr ThrowingMove operator()(int) const noexcept
  {
    return ThrowingMove{};
  }
};

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  int a[]    = {1, 2, 3, 4};
  double b[] = {0.1, 0.2, 0.3};

  {
    cuda::zip_iterator iter1{a, b};
    cuda::zip_iterator iter2{a + 1, b + 1};

    cuda::std::ranges::iter_swap(iter1, iter2);

    assert(a[0] == 2);
    assert(a[1] == 1);
    assert(b[0] == 0.2);
    assert(b[1] == 0.1);

    auto [x1, y1] = *iter1;
    assert(cuda::std::addressof(x1) == cuda::std::addressof(a[0]));
    assert(cuda::std::addressof(y1) == cuda::std::addressof(b[0]));

    auto [x2, y2] = *iter2;
    assert(cuda::std::addressof(x2) == cuda::std::addressof(a[1]));
    assert(cuda::std::addressof(y2) == cuda::std::addressof(b[1]));

    static_assert(noexcept(cuda::std::ranges::iter_swap(iter1, iter2)));
  }

  { // We need an iterator whose rvalue reference is potentially throwing on move construction
    ThrowingMove throwing[3] = {ThrowingMove{}, ThrowingMove{}, ThrowingMove{}};
    cuda::zip_iterator iter1{throwing};
    [[maybe_unused]] cuda::zip_iterator iter2{throwing};
    static_assert(!noexcept(cuda::std::ranges::iter_swap(iter1, iter2)));
  }

  { // underlying iterators iter_swap are called through ranges::iter_swap
    int iter_move_called_times1 = 0;
    int iter_move_called_times2 = 0;
    int iter_swap_called_times1 = 0;
    int iter_swap_called_times2 = 0;

    using Iter = cuda::zip_iterator<adltest::iter_move_swap_iterator, adltest::iter_move_swap_iterator>;
    Iter iter1{{iter_move_called_times1, iter_swap_called_times1, 0},
               {iter_move_called_times2, iter_swap_called_times2, 0}};
    Iter iter2 = cuda::std::ranges::next(iter1, 3);

    assert(iter_move_called_times1 == 0);
    assert(iter_move_called_times2 == 0);
    assert(iter_swap_called_times1 == 0);
    assert(iter_swap_called_times2 == 0);

    cuda::std::ranges::iter_swap(iter1, iter2);
    assert(iter_move_called_times1 == 0);
    assert(iter_move_called_times2 == 0);
    assert(iter_swap_called_times1 == 2);
    assert(iter_swap_called_times2 == 2);

    cuda::std::ranges::iter_swap(iter1, iter2);
    assert(iter_move_called_times1 == 0);
    assert(iter_move_called_times2 == 0);
    assert(iter_swap_called_times1 == 4);
    assert(iter_swap_called_times2 == 4);
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2020
  static_assert(test());
#endif // TEST_STD_VER >= 2020

  return 0;
}
