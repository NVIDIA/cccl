//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class Alloc>
//   tuple(allocator_arg_t, const Alloc& a, const Types&...);

#include <cuda/std/__memory_>
#include <cuda/std/cassert>
#include <cuda/std/tuple>

#include "../alloc_constexpr_types.h"
#include "allocators.h"
#include "test_macros.h"

struct ImplicitCopy
{
  TEST_FUNC explicit ImplicitCopy(int) {}
  TEST_FUNC ImplicitCopy(ImplicitCopy const&) {}
};

// Test that tuple(cuda::std::allocator_arg, Alloc, Types const&...) allows implicit
// copy conversions in return value expressions.
TEST_FUNC cuda::std::tuple<ImplicitCopy> testImplicitCopy1()
{
  ImplicitCopy i(42);
  return {cuda::std::allocator_arg, cuda::std::allocator<void>{}, i};
}

TEST_FUNC cuda::std::tuple<ImplicitCopy> testImplicitCopy2()
{
  const ImplicitCopy i(42);
  return {cuda::std::allocator_arg, cuda::std::allocator<void>{}, i};
}

TEST_FUNC constexpr bool test()
{
  A1<int> alloc{5};
  {
    const int v = 3;
    cuda::std::tuple<constexpr_alloc_arg> t(cuda::std::allocator_arg, alloc, v);
    assert(cuda::std::get<0>(t).value == 3);
  }
  {
    const int a = 1;
    const int b = 2;
    cuda::std::tuple<constexpr_alloc_arg, constexpr_alloc_last> t(cuda::std::allocator_arg, alloc, a, b);
    assert(cuda::std::get<0>(t).value == 1);
    assert(cuda::std::get<1>(t).value == 2);
  }
  {
    const int v = 3;
    cuda::std::tuple<int> t(cuda::std::allocator_arg, alloc, v);
    assert(cuda::std::get<0>(t) == 3);
  }
  {
    const constexpr_alloc_arg src(cuda::std::allocator_arg, alloc, 3);
    cuda::std::tuple<constexpr_alloc_arg> t(cuda::std::allocator_arg, alloc, src);
    assert(cuda::std::get<0>(t).value == 3);
  }
  {
    const constexpr_alloc_last src(3, alloc);
    cuda::std::tuple<constexpr_alloc_last> t(cuda::std::allocator_arg, alloc, src);
    assert(cuda::std::get<0>(t).value == 3);
  }
  {
    const int a = 10;
    const constexpr_alloc_arg b(cuda::std::allocator_arg, alloc, 15);
    cuda::std::tuple<int, constexpr_alloc_arg> t(cuda::std::allocator_arg, alloc, a, b);
    assert(cuda::std::get<0>(t) == 10);
    assert(cuda::std::get<1>(t).value == 15);
  }
  {
    const int a = 1;
    const constexpr_alloc_arg b(cuda::std::allocator_arg, alloc, 2);
    const constexpr_alloc_last c(3, alloc);
    cuda::std::tuple<int, constexpr_alloc_arg, constexpr_alloc_last> t(cuda::std::allocator_arg, alloc, a, b, c);
    assert(cuda::std::get<0>(t) == 1);
    assert(cuda::std::get<1>(t).value == 2);
    assert(cuda::std::get<2>(t).value == 3);
  }
  {
    // A2 is not convertible to the element allocator type, so ordinary copy is selected.
    const int a = 1;
    const constexpr_alloc_arg b(cuda::std::allocator_arg, alloc, 2);
    const constexpr_alloc_last c(3, alloc);
    cuda::std::tuple<int, constexpr_alloc_arg, constexpr_alloc_last> t(cuda::std::allocator_arg, A2<int>{5}, a, b, c);
    assert(cuda::std::get<0>(t) == 1);
    assert(cuda::std::get<1>(t).value == -1);
    assert(cuda::std::get<2>(t).value == -1);
  }
  return true;
}

TEST_FUNC void test_runtime()
{
  // cuda::std::allocator is not constexpr in C++17.
  // check that the literal '0' can implicitly initialize a stored pointer.
  [[maybe_unused]] cuda::std::tuple<int*> t = {cuda::std::allocator_arg, cuda::std::allocator<void>{}, 0};
}

int main(int, char**)
{
  test();
  static_assert(test());
  test_runtime();
  return 0;
}
