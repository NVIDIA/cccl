//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// ranges

// inline constexpr unspecified indices = unspecified;

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"
#include "type_algorithms.h"
#include "types.h"

// Test SFINAE.

template <class SizeType, class = void>
inline constexpr bool HasIndices = false;
template <class SizeType>
inline constexpr bool
  HasIndices<SizeType, cuda::std::void_t<decltype(cuda::std::ranges::views::indices(cuda::std::declval<SizeType>()))>> =
    true;

struct HasIndicesFn
{
  template <class T>
  TEST_FUNC void operator()() const
  {
    static_assert(HasIndices<T>);
  }
};

struct NotIntegerLike
{};

TEST_FUNC void test_SFINAE()
{
  static_assert(HasIndices<cuda::std::size_t>);
  types::for_each(types::integer_types(), HasIndicesFn{});

  // Non-integer-like types should not satisfy HasIndices
  static_assert(!HasIndices<bool>);
  static_assert(!HasIndices<float>);
  static_assert(!HasIndices<void>);
  static_assert(!HasIndices<SomeInt>); // Does satisfy is_integer_like, but not the conversion to cuda::std::size_t
  static_assert(!HasIndices<NotIntegerLike>);
}

TEST_FUNC constexpr bool test()
{
  {
    auto indices_view = cuda::std::ranges::views::indices(5);
    static_assert(cuda::std::ranges::range<decltype(indices_view)>);

    assert(indices_view.size() == 5);

    assert(indices_view[0] == 0);
    assert(indices_view[1] == 1);
    assert(indices_view[2] == 2);
    assert(indices_view[3] == 3);
    assert(indices_view[4] == 4);
  }

  {
    cuda::std::array<int, 5> v{1, 2, 3, 4, 5};

    auto indices_view = cuda::std::ranges::views::indices(cuda::std::ranges::size(v));
    static_assert(cuda::std::ranges::range<decltype(indices_view)>);

    assert(indices_view.size() == 5);

    assert(indices_view[0] == 0);
    assert(indices_view[1] == 1);
    assert(indices_view[2] == 2);
    assert(indices_view[3] == 3);
    assert(indices_view[4] == 4);
  }

  {
    cuda::std::array<SomeInt, 5> v{SomeInt{1}, SomeInt{2}, SomeInt{3}, SomeInt{4}, SomeInt{5}};

    auto indices_view = cuda::std::ranges::views::indices(cuda::std::ranges::size(v));
    static_assert(cuda::std::ranges::range<decltype(indices_view)>);

    assert(indices_view.size() == 5);

    assert(indices_view[0] == 0);
    assert(indices_view[1] == 1);
    assert(indices_view[2] == 2);
    assert(indices_view[3] == 3);
    assert(indices_view[4] == 4);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
