//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, msvc

// XFAIL: gcc-4

// <utility>

// template <class T, class U> struct pair;

// pair(pair const&) = default;
// pair(pair &&) = default;
// pair& operator=(pair const&);
// pair& operator=(pair&&);

// Test that the copy/move constructors and assignment operators are
// correctly defined or deleted based on the properties of `T` and `U`.

#include <cuda/std/cassert>
// cuda::std::string not supported
// #include <cuda/std/string>
#include <cuda/std/tuple>

#include "archetypes.h"
#include "test_macros.h"
using namespace ImplicitTypes; // Get implicitly archetypes

namespace ConstructorTest
{

template <class T1, bool CanCopy = true, bool CanMove = CanCopy>
__host__ __device__ void test()
{
  using P1 = cuda::std::pair<T1, int>;
  using P2 = cuda::std::pair<int, T1>;
  static_assert(cuda::std::is_copy_constructible<P1>::value == CanCopy, "");
  static_assert(cuda::std::is_move_constructible<P1>::value == CanMove, "");
  static_assert(cuda::std::is_copy_constructible<P2>::value == CanCopy, "");
  static_assert(cuda::std::is_move_constructible<P2>::value == CanMove, "");
};

} // namespace ConstructorTest

__host__ __device__ void test_constructors_exist()
{
  using namespace ConstructorTest;
  {
    test<int>();
    test<int&>();
    test<int&&, false, true>();
    test<const int>();
    test<const int&>();
    test<const int&&, false, true>();
  }
  {
    test<Copyable>();
    test<Copyable&>();
    test<Copyable&&, false, true>();
  }
  {
    test<NonCopyable, false>();
    test<NonCopyable&, true>();
    test<NonCopyable&&, false, true>();
  }
  {
    // Even though CopyOnly has an explicitly deleted move constructor
    // pair's move constructor is only implicitly deleted and therefore
    // it doesn't participate in overload resolution.
    test<CopyOnly, true, true>();
    test<CopyOnly&, true>();
    test<CopyOnly&&, false, true>();
  }
  {
    test<MoveOnly, false, true>();
    test<MoveOnly&, true>();
    test<MoveOnly&&, false, true>();
  }
}

namespace AssignmentOperatorTest
{

template <class T1, bool CanCopy = true, bool CanMove = CanCopy>
__host__ __device__ void test()
{
  using P1 = cuda::std::pair<T1, int>;
  using P2 = cuda::std::pair<int, T1>;
  static_assert(cuda::std::is_copy_assignable<P1>::value == CanCopy, "");
  static_assert(cuda::std::is_move_assignable<P1>::value == CanMove, "");
  static_assert(cuda::std::is_copy_assignable<P2>::value == CanCopy, "");
  static_assert(cuda::std::is_move_assignable<P2>::value == CanMove, "");
};

} // namespace AssignmentOperatorTest

__host__ __device__ void test_assignment_operator_exists()
{
  using namespace AssignmentOperatorTest;
  {
    test<int>();
    test<int&>();
    test<int&&>();
    test<const int, false>();
    test<const int&, false>();
    test<const int&&, false>();
  }
  {
    test<Copyable>();
    test<Copyable&>();
    test<Copyable&&>();
  }
  {
    test<NonCopyable, false>();
    test<NonCopyable&, false>();
    test<NonCopyable&&, false>();
  }
  {
    test<CopyOnly, true>();
    test<CopyOnly&, true>();
    test<CopyOnly&&, true>();
  }
  {
    test<MoveOnly, false, true>();
    test<MoveOnly&, false, false>();
    test<MoveOnly&&, false, true>();
  }
}

int main(int, char**)
{
  test_constructors_exist();
  test_assignment_operator_exists();

  return 0;
}
