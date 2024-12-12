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

// constexpr explicit repeat_view(const T& value, Bound bound = Bound()) requires copy_constructible<T>;
// constexpr explicit repeat_view(T&& value, Bound bound = Bound());

#include <cuda/std/cassert>
#include <cuda/std/iterator>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>

#include "MoveOnly.h"
#include "test_macros.h"

struct Empty
{};

// Test explicit
static_assert(cuda::std::is_constructible_v<cuda::std::ranges::repeat_view<Empty>, const Empty&>);
static_assert(cuda::std::is_constructible_v<cuda::std::ranges::repeat_view<Empty>, Empty&&>);
static_assert(cuda::std::is_constructible_v<cuda::std::ranges::repeat_view<Empty, int>, const Empty&>);
static_assert(cuda::std::is_constructible_v<cuda::std::ranges::repeat_view<Empty, int>, Empty&&>);

static_assert(!cuda::std::is_convertible_v<const Empty&, cuda::std::ranges::repeat_view<Empty>>);
static_assert(!cuda::std::is_convertible_v<Empty&&, cuda::std::ranges::repeat_view<Empty>>);
static_assert(!cuda::std::is_convertible_v<const Empty&, cuda::std::ranges::repeat_view<Empty, int>>);
static_assert(!cuda::std::is_convertible_v<Empty&&, cuda::std::ranges::repeat_view<Empty, int>>);

static_assert(!cuda::std::is_constructible_v<cuda::std::ranges::repeat_view<MoveOnly>, const MoveOnly&>);
static_assert(cuda::std::is_constructible_v<cuda::std::ranges::repeat_view<MoveOnly>, MoveOnly&&>);

__host__ __device__ constexpr bool test()
{
  // Move && unbound && default argument
  {
    cuda::std::ranges::repeat_view<Empty> rv(Empty{});
    assert(rv.begin() + 10 != rv.end());
  }

  // Move && unbound && user-provided argument
  {
    cuda::std::ranges::repeat_view<Empty> rv(Empty{}, cuda::std::unreachable_sentinel);
    assert(rv.begin() + 10 != rv.end());
  }

  // Move && bound && default argument
  {
    cuda::std::ranges::repeat_view<Empty, int> rv(Empty{});
    assert(rv.begin() == rv.end());
  }

  // Move && bound && user-provided argument
  {
    cuda::std::ranges::repeat_view<Empty, int> rv(Empty{}, 10);
    assert(rv.begin() + 10 == rv.end());
  }

  // Copy && unbound && default argument
  {
    Empty e{};
    cuda::std::ranges::repeat_view<Empty> rv(e);
    assert(rv.begin() + 10 != rv.end());
  }

  // Copy && unbound && user-provided argument
  {
    Empty e{};
    cuda::std::ranges::repeat_view<Empty> rv(e, cuda::std::unreachable_sentinel);
    assert(rv.begin() + 10 != rv.end());
  }

  // Copy && bound && default argument
  {
    Empty e{};
    cuda::std::ranges::repeat_view<Empty, int> rv(e);
    assert(rv.begin() == rv.end());
  }

  // Copy && bound && user-provided argument
  {
    Empty e{};
    cuda::std::ranges::repeat_view<Empty, int> rv(e, 10);
    assert(rv.begin() + 10 == rv.end());
  }

  return true;
}

int main(int, char**)
{
  test();
#if !defined(TEST_COMPILER_CLANG) || __clang__ > 9
  static_assert(test());
#endif // clang > 9

  return 0;
}
