//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/array>
#include <cuda/std/cstddef>
#include <cuda/std/initializer_list>
#include <cuda/std/ranges>
#include <cuda/std/span>
#include <cuda/std/type_traits>

#include "test_macros.h"

struct NonDefaultConstructible
{
  NonDefaultConstructible() = delete;
};

struct MemberSize
{
  MemberSize() = delete;

  TEST_FUNC constexpr unsigned short size()
  {
    return 42;
  }
};

struct ADLSize
{
  TEST_FUNC friend constexpr int size(ADLSize&)
  {
    return 7;
  }
};

struct RuntimeSize
{
  TEST_FUNC cuda::std::size_t size() const
  {
    return 3;
  }
};

struct ValueDependentSize
{
  cuda::std::size_t count = 0;

  TEST_FUNC constexpr cuda::std::size_t size() const
  {
    return count;
  }
};

struct SentinelSize
{
  int values[3];

  TEST_FUNC constexpr int* begin()
  {
    return values;
  }

  TEST_FUNC constexpr int* end()
  {
    return values + 3;
  }
};

struct DisabledSize : MemberSize
{};

template <>
inline constexpr bool cuda::std::ranges::disable_sized_range<DisabledSize> = true;

template <class Range, cuda::std::size_t Expected>
TEST_FUNC constexpr void test_cvref()
{
  static_assert(cuda::std::ranges::__static_size_v<Range> == Expected);
  static_assert(cuda::std::ranges::__static_size_v<Range&> == Expected);
  static_assert(cuda::std::ranges::__static_size_v<Range&&> == Expected);
  static_assert(cuda::std::ranges::__static_size_v<const Range> == Expected);
  static_assert(cuda::std::ranges::__static_size_v<const Range&> == Expected);
  static_assert(cuda::std::ranges::__static_size_v<const Range&&> == Expected);
  static_assert(cuda::std::ranges::__static_size_v<volatile Range> == Expected);
  static_assert(cuda::std::ranges::__static_size_v<const volatile Range&> == Expected);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::ranges::__static_size_v<Range>), const cuda::std::size_t>);
}

TEST_FUNC constexpr bool test()
{
  test_cvref<int[3], 3>();
  test_cvref<int[2][3], 2>();
  test_cvref<NonDefaultConstructible[4], 4>();
  test_cvref<cuda::std::array<int, 0>, 0>();
  test_cvref<cuda::std::array<int, 5>, 5>();
  test_cvref<cuda::std::array<NonDefaultConstructible, 5>, 5>();
  test_cvref<cuda::std::span<int, 0>, 0>();
  test_cvref<cuda::std::span<int, 6>, 6>();
  test_cvref<cuda::std::span<NonDefaultConstructible, 6>, 6>();
  test_cvref<MemberSize, 42>();
  test_cvref<ADLSize, 7>();
  test_cvref<SentinelSize, 3>();

  test_cvref<cuda::std::span<int>, cuda::std::dynamic_extent>();
  test_cvref<cuda::std::initializer_list<int>, cuda::std::dynamic_extent>();
  test_cvref<RuntimeSize, cuda::std::dynamic_extent>();
  test_cvref<ValueDependentSize, cuda::std::dynamic_extent>();
  test_cvref<DisabledSize, cuda::std::dynamic_extent>();
  test_cvref<int[], cuda::std::dynamic_extent>();
  test_cvref<int*, cuda::std::dynamic_extent>();
  test_cvref<int, cuda::std::dynamic_extent>();
  static_assert(cuda::std::ranges::__static_size_v<void> == cuda::std::dynamic_extent);
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
