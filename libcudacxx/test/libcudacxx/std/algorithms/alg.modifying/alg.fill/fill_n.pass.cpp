//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<class Iter, IntegralLike Size, class T>
//   requires OutputIterator<Iter, const T&>
//   constexpr OutputIterator      // constexpr after C++17
//   fill_n(Iter first, Size n, const T& value);

#include <cuda/std/algorithm>
#include <cuda/std/cassert>

#include "test_iterators.h"
#include "test_macros.h"
#include "user_defined_integral.h"

using UDI = UserDefinedIntegral<unsigned>;

template <class Iter>
TEST_FUNC constexpr void test_char()
{
  char a[4] = {};
  Iter it   = cuda::std::fill_n(Iter(a), UDI(4), char(1));
  assert(base(it) == a + 4);
  assert(a[0] == 1);
  assert(a[1] == 1);
  assert(a[2] == 1);
  assert(a[3] == 1);
}

template <class Iter>
TEST_FUNC constexpr void test_int()
{
  int a[4] = {};
  Iter it  = cuda::std::fill_n(Iter(a), UDI(4), 1);
  assert(base(it) == a + 4);
  assert(a[0] == 1);
  assert(a[1] == 1);
  assert(a[2] == 1);
  assert(a[3] == 1);
}

TEST_FUNC constexpr void test_int_array()
{
  int a[4] = {};
  assert(cuda::std::fill_n(a, UDI(4), static_cast<char>(1)) == a + 4);
  assert(a[0] == 1);
  assert(a[1] == 1);
  assert(a[2] == 1);
  assert(a[3] == 1);
}

// nvcc does not support modification of mutable variables in a constexpr context
struct source
{
  TEST_FUNC source()
      : i(0)
  {}

  TEST_FUNC operator int() const
  {
    return i++;
  }
  mutable int i;
};

TEST_FUNC void test_int_array_struct_source()
{
  int a[4] = {};
  assert(cuda::std::fill_n(a, UDI(4), source()) == a + 4);
  assert(a[0] == 0);
  assert(a[1] == 1);
  assert(a[2] == 2);
  assert(a[3] == 3);
}

struct test1
{
  TEST_FUNC constexpr test1()
      : c(0)
  {}
  TEST_FUNC constexpr test1(char xc)
      : c(xc + 1)
  {}
  char c;
};

TEST_FUNC constexpr void test_struct_array()
{
  test1 test1a[4] = {};
  assert(cuda::std::fill_n(test1a, UDI(4), static_cast<char>(10)) == test1a + 4);
  assert(test1a[0].c == 11);
  assert(test1a[1].c == 11);
  assert(test1a[2].c == 11);
  assert(test1a[3].c == 11);
}

class A
{
  char a_;

public:
  TEST_FUNC constexpr A()
      : a_()
  {}
  TEST_FUNC explicit constexpr A(char a)
      : a_(a)
  {}
  TEST_FUNC constexpr operator unsigned char() const
  {
    return 'b';
  }

  TEST_FUNC constexpr friend bool operator==(const A& x, const A& y)
  {
    return x.a_ == y.a_;
  }
};

TEST_FUNC constexpr void test5()
{
  A a[3];
  assert(cuda::std::fill_n(&a[0], UDI(3), A('a')) == a + 3);
  assert(a[0] == A('a'));
  assert(a[1] == A('a'));
  assert(a[2] == A('a'));
}

struct Storage
{
  union
  {
    unsigned char a;
    unsigned char b;
  };
};

TEST_FUNC constexpr void test6()
{
  Storage foo[5] = {};
  cuda::std::fill_n(&foo[0], UDI(5), Storage());
}

TEST_FUNC constexpr bool test()
{
  test_char<cpp17_output_iterator<char*>>();
  test_char<forward_iterator<char*>>();
  test_char<bidirectional_iterator<char*>>();
  test_char<random_access_iterator<char*>>();
  test_char<char*>();

  test_int<cpp17_output_iterator<int*>>();
  test_int<forward_iterator<int*>>();
  test_int<bidirectional_iterator<int*>>();
  test_int<random_access_iterator<int*>>();
  test_int<int*>();

  test_int_array();
  test_struct_array();

  test5();
  test6();

#if !TEST_COMPILER(NVRTC)
  NV_IF_TARGET(NV_IS_HOST, (test_int<host_only_iterator<int*>>();))
#endif // !TEST_COMPILER(NVRTC)
#if TEST_CUDA_COMPILATION()
  NV_IF_TARGET(NV_IS_DEVICE, (test_int<device_only_iterator<int*>>();))
#endif // TEST_CUDA_COMPILATION()

  return true;
}

int main(int, char**)
{
  test();
  test_int_array_struct_source();
  static_assert(test());

  return 0;
}
