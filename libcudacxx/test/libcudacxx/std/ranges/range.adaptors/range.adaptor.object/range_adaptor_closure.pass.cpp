//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// cuda::std::ranges::range_adaptor_closure
// operator| composition of two range adaptor closures

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/utility>

#include "test_macros.h"

// Adds `offset_` to a range sum, or to an integer produced by another closure.
// The move constructor is not constexpr and may throw, while the copy constructor does not.
struct ThrowingMoveClosure : cuda::std::ranges::range_adaptor_closure<ThrowingMoveClosure>
{
  int offset_ = 0;

  TEST_FUNC constexpr ThrowingMoveClosure() noexcept
      : offset_(0)
  {}
  TEST_FUNC constexpr explicit ThrowingMoveClosure(int offset) noexcept
      : offset_(offset)
  {}
  TEST_FUNC constexpr ThrowingMoveClosure(const ThrowingMoveClosure& other) noexcept
      : offset_(other.offset_)
  {}
  TEST_FUNC ThrowingMoveClosure(ThrowingMoveClosure&& other) noexcept(false)
      : offset_(other.offset_)
  {}

  template <class Range>
  TEST_FUNC constexpr int operator()(Range&& range) const
  {
    int sum = offset_;
    for (int value : range)
    {
      sum += value;
    }
    return sum;
  }

  TEST_FUNC constexpr int operator()(int value) const
  {
    return value + offset_;
  }
};

// Copyable closure whose move constructor is deleted. Composition must copy it into place.
struct CopyOnlyClosure : cuda::std::ranges::range_adaptor_closure<CopyOnlyClosure>
{
  int offset_ = 0;

  TEST_FUNC constexpr CopyOnlyClosure() noexcept
      : offset_(0)
  {}
  TEST_FUNC constexpr explicit CopyOnlyClosure(int offset) noexcept
      : offset_(offset)
  {}
  TEST_FUNC constexpr CopyOnlyClosure(const CopyOnlyClosure& other) noexcept
      : offset_(other.offset_)
  {}
  CopyOnlyClosure(CopyOnlyClosure&&) = delete;

  template <class Range>
  TEST_FUNC constexpr int operator()(Range&& range) const
  {
    int sum = offset_;
    for (int value : range)
    {
      sum += value;
    }
    return sum;
  }

  TEST_FUNC constexpr int operator()(int value) const
  {
    return value + offset_;
  }
};

TEST_FUNC constexpr bool test()
{
  int buf[] = {1, 2, 3};
  ThrowingMoveClosure first{1};
  ThrowingMoveClosure second{10};

  static_assert(noexcept(first | second));
  auto composed = first | second;
  // 1 + 2 + 3 + 1, then + 10. Invoke the lvalue so the stored closures are copied, not moved.
  assert(composed(buf) == 17);
  assert(second(first(buf)) == 17);

  CopyOnlyClosure copy_first{1};
  CopyOnlyClosure copy_second{10};
  static_assert(noexcept(copy_first | copy_second));
  auto copy_composed = copy_first | copy_second;
  assert(copy_composed(buf) == 17);
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
