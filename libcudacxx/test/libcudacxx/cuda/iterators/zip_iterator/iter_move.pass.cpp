//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// friend constexpr auto iter_move(const iterator& i) noexcept(see below);

#include <cuda/iterator>
#include <cuda/std/cassert>
#include <cuda/std/iterator>
#include <cuda/std/tuple>

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

struct ThrowingMove
{
  __host__ __device__ constexpr ThrowingMove() noexcept {}
  __host__ __device__ constexpr ThrowingMove(ThrowingMove&&) noexcept(false) {}
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
  int a[]          = {1, 2, 3, 4};
  const double b[] = {3.0, 4.0};

  { // underlying iter_move noexcept
    cuda::zip_iterator iter{a, b, cuda::counting_iterator{3L}};

    assert(cuda::std::ranges::iter_move(iter) == cuda::std::make_tuple(1, 3.0, 3L));
    static_assert(
      cuda::std::is_same_v<decltype(cuda::std::ranges::iter_move(iter)), cuda::std::tuple<int&&, const double&&, long>>);
    static_assert(noexcept(cuda::std::ranges::iter_move(iter)));
  }

  { // We need an iterator whose rvalue reference is potentially throwing on move construction
    [[maybe_unused]] cuda::zip_iterator iter{cuda::transform_iterator{cuda::counting_iterator{0}, ToThrowingMove{}}};
    static_assert(!noexcept(cuda::std::ranges::iter_move(iter)));
  }

  { // underlying iterators' iter_move are called through ranges::iter_move
    int iter_move_called_times1 = 0;
    int iter_move_called_times2 = 0;
    int iter_swap_called_times1 = 0;
    int iter_swap_called_times2 = 0;

    using Iter = cuda::zip_iterator<adltest::iter_move_swap_iterator, adltest::iter_move_swap_iterator>;
    Iter iter{{iter_move_called_times1, iter_swap_called_times1, 0},
              {iter_move_called_times2, iter_swap_called_times2, 0}};
    assert(iter_move_called_times1 == 0);
    assert(iter_move_called_times2 == 0);
    assert(iter_swap_called_times1 == 0);
    assert(iter_swap_called_times2 == 0);
    {
      [[maybe_unused]] auto&& i = cuda::std::ranges::iter_move(iter);
      assert(iter_move_called_times1 == 1);
      assert(iter_move_called_times2 == 1);
      assert(iter_swap_called_times1 == 0);
      assert(iter_swap_called_times2 == 0);
    }
    {
      [[maybe_unused]] auto&& i = cuda::std::ranges::iter_move(iter);
      assert(iter_move_called_times1 == 2);
      assert(iter_move_called_times2 == 2);
      assert(iter_swap_called_times1 == 0);
      assert(iter_swap_called_times2 == 0);
    }
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
