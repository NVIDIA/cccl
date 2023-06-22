//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//

#include <cuda/std/type_traits>
#include <cuda/std/cassert>

#include "test_macros.h"

struct Bomb;
template <int N, class T = Bomb >
struct BOOM {
  using Explode = typename T::BOOMBOOM;
};

using True = cuda::std::true_type;
using False = cuda::std::false_type;

__host__ __device__
void test_if() {
  ASSERT_SAME_TYPE(cuda::std::_If<true, int, long>, int);
  ASSERT_SAME_TYPE(cuda::std::_If<false, int, long>, long);
}

__host__ __device__
void test_and() {
  static_assert(cuda::std::_And<True>::value, "");
  static_assert(!cuda::std::_And<False>::value, "");
  static_assert(cuda::std::_And<True, True>::value, "");
  static_assert(!cuda::std::_And<False, BOOM<1> >::value, "");
  static_assert(!cuda::std::_And<True, True, True, False, BOOM<2> >::value, "");
}

__host__ __device__
void test_or() {
  static_assert(cuda::std::_Or<True>::value, "");
  static_assert(!cuda::std::_Or<False>::value, "");
  static_assert(cuda::std::_Or<False, True>::value, "");
  static_assert(cuda::std::_Or<True, cuda::std::_Not<BOOM<3> > >::value, "");
  static_assert(!cuda::std::_Or<False, False>::value, "");
  static_assert(cuda::std::_Or<True, BOOM<1> >::value, "");
  static_assert(cuda::std::_Or<False, False, False, False, True, BOOM<2> >::value, "");
}

__host__ __device__
void test_combined() {
  static_assert(cuda::std::_And<True, cuda::std::_Or<False, True, BOOM<4> > >::value, "");
  static_assert(cuda::std::_And<True, cuda::std::_Or<False, True, BOOM<4> > >::value, "");
  static_assert(cuda::std::_Not<cuda::std::_And<True, False, BOOM<5> > >::value, "");
}

struct MemberTest {
  static const int foo;
  using type = long;

  __host__ __device__
  void func(int);
};
struct Empty {};
struct MemberTest2 {
  using foo = int;
};
template <class T>
using HasFooData = decltype(T::foo);
template <class T>
using HasFooType = typename T::foo;

template <class T, class U>
using FuncCallable = decltype(cuda::std::declval<T>().func(cuda::std::declval<U>()));
template <class T>
using BadCheck = typename T::DOES_NOT_EXIST;

__host__ __device__
void test_is_valid_trait() {
  static_assert(cuda::std::_IsValidExpansion<HasFooData, MemberTest>::value, "");
  static_assert(!cuda::std::_IsValidExpansion<HasFooType, MemberTest>::value, "");
  static_assert(!cuda::std::_IsValidExpansion<HasFooData, MemberTest2>::value, "");
  static_assert(cuda::std::_IsValidExpansion<HasFooType, MemberTest2>::value, "");
  static_assert(cuda::std::_IsValidExpansion<FuncCallable, MemberTest, int>::value, "");
  static_assert(!cuda::std::_IsValidExpansion<FuncCallable, MemberTest, void*>::value, "");
}

__host__ __device__
void test_first_and_second_type() {
  ASSERT_SAME_TYPE(cuda::std::_FirstType<int, long, void*>, int);
  ASSERT_SAME_TYPE(cuda::std::_FirstType<char>, char);
  ASSERT_SAME_TYPE(cuda::std::_SecondType<char, long>, long);
  ASSERT_SAME_TYPE(cuda::std::_SecondType<long long, int, void*>, int);
}

int main(int, char**) {
  return 0;
}
