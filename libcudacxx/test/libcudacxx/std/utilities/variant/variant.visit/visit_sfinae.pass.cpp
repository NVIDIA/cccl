//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: msvc-19.16
// UNSUPPORTED: clang-7, clang-8

// <cuda/std/variant>
// template <class Visitor, class... Variants>
// constexpr see below visit(Visitor&& vis, Variants&&... vars);

#include <cuda/std/cassert>
// #include <cuda/std/memory>
// #include <cuda/std/string>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/std/variant>

#include "test_macros.h"
#include "variant_test_helpers.h"

struct any_visitor
{
  template <typename T>
  __host__ __device__ void operator()(const T&) const
  {}
};

template <typename T, typename = decltype(cuda::std::visit(cuda::std::declval<any_visitor&>(), cuda::std::declval<T>()))>
__host__ __device__ constexpr bool has_visit(int)
{
  return true;
}

template <typename T>
__host__ __device__ constexpr bool has_visit(...)
{
  return false;
}

__host__ __device__ void test_sfinae()
{
  struct BadVariant
      : cuda::std::variant<short>
      , cuda::std::variant<long, float>
  {};
  struct BadVariant2 : private cuda::std::variant<long, float>
  {};
  struct GoodVariant : cuda::std::variant<long, float>
  {};
  struct GoodVariant2 : GoodVariant
  {};

  static_assert(!has_visit<int>(0), "");
#if !defined(TEST_COMPILER_MSVC) // MSVC cannot deal with that even with std::variant
  static_assert(!has_visit<BadVariant>(0), "");
#endif // !TEST_COMPILER_MSVC
  static_assert(!has_visit<BadVariant2>(0), "");
  static_assert(has_visit<cuda::std::variant<int>>(0), "");
  static_assert(has_visit<GoodVariant>(0), "");
  static_assert(has_visit<GoodVariant2>(0), "");
}

int main(int, char**)
{
  test_sfinae();

  return 0;
}
