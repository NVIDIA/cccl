//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: enable-tile
// nvbug6067464: error: Internal Compiler Error (tile codegen): "call to unknown tile builtin function!

// UNSUPPORTED: msvc-19.16
// UNSUPPORTED: clang-7, clang-8

// <cuda/std/variant>
// template <class R, class Visitor, class... Variants>
// constexpr R visit(Visitor&& vis, Variants&&... vars);

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
  TEST_FUNC bool operator()(const T&)
  {
    return true;
  }
};

template <class T>
_CCCL_CONCEPT has_visit =
  _CCCL_REQUIRES_EXPR((T), T&& t)((cuda::std::visit<bool>(any_visitor{}, cuda::std::forward<T>(t))));

TEST_FUNC void test_sfinae()
{
  struct BadVariant
      : cuda::std::variant<short>
      , cuda::std::variant<long, float>
  {};

  static_assert(has_visit<cuda::std::variant<int>>);
#if !TEST_COMPILER(MSVC) // MSVC cannot deal with that even with std::variant
  static_assert(!has_visit<BadVariant>);
#endif // !TEST_COMPILER(MSVC)
}

int main(int, char**)
{
  test_sfinae();

  return 0;
}
