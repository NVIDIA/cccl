//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Test that iota_view conforms to range and view concepts.

#include <cuda/std/ranges>

#include "test_macros.h"
#include "types.h"

struct Decrementable
{
  using difference_type = int;

#if TEST_HAS_SPACESHIP()
  auto operator<=>(const Decrementable&) const = default;
#else
  TEST_FUNC bool operator==(const Decrementable&) const;
  TEST_FUNC bool operator!=(const Decrementable&) const;

  TEST_FUNC bool operator<(const Decrementable&) const;
  TEST_FUNC bool operator<=(const Decrementable&) const;
  TEST_FUNC bool operator>(const Decrementable&) const;
  TEST_FUNC bool operator>=(const Decrementable&) const;
#endif

  TEST_FUNC Decrementable& operator++();
  TEST_FUNC Decrementable operator++(int);
  TEST_FUNC Decrementable& operator--();
  TEST_FUNC Decrementable operator--(int);
};

struct Incrementable
{
  using difference_type = int;

#if TEST_HAS_SPACESHIP()
  auto operator<=>(const Incrementable&) const = default;
#else
  TEST_FUNC bool operator==(const Incrementable&) const;
  TEST_FUNC bool operator!=(const Incrementable&) const;

  TEST_FUNC bool operator<(const Incrementable&) const;
  TEST_FUNC bool operator<=(const Incrementable&) const;
  TEST_FUNC bool operator>(const Incrementable&) const;
  TEST_FUNC bool operator>=(const Incrementable&) const;
#endif

  TEST_FUNC Incrementable& operator++();
  TEST_FUNC Incrementable operator++(int);
};

static_assert(cuda::std::ranges::random_access_range<cuda::std::ranges::iota_view<int>>);
static_assert(cuda::std::ranges::random_access_range<const cuda::std::ranges::iota_view<int>>);
static_assert(cuda::std::ranges::bidirectional_range<cuda::std::ranges::iota_view<Decrementable>>);
static_assert(cuda::std::ranges::forward_range<cuda::std::ranges::iota_view<Incrementable>>);
static_assert(cuda::std::ranges::input_range<cuda::std::ranges::iota_view<NotIncrementable>>);
static_assert(cuda::std::ranges::view<cuda::std::ranges::iota_view<int>>);

int main(int, char**)
{
  return 0;
}
