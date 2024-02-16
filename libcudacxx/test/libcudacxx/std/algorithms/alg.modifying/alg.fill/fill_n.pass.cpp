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

#include <cuda/std/__algorithm>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "test_iterators.h"
#include "user_defined_integral.h"

using UDI = UserDefinedIntegral<unsigned>;


template <class Iter>
TEST_HOST_DEVICE TEST_CONSTEXPR_CXX14 void
test_char()
{
    char a[4] = {};
    Iter it = cuda::std::fill_n(Iter(a), UDI(4), char(1));
    assert(base(it) == a + 4);
    assert(a[0] == 1);
    assert(a[1] == 1);
    assert(a[2] == 1);
    assert(a[3] == 1);
}

template <class Iter>
TEST_HOST_DEVICE TEST_CONSTEXPR_CXX14 void
test_int()
{
    int a[4] = {};
    Iter it = cuda::std::fill_n(Iter(a), UDI(4), 1);
    assert(base(it) == a + 4);
    assert(a[0] == 1);
    assert(a[1] == 1);
    assert(a[2] == 1);
    assert(a[3] == 1);
}

TEST_HOST_DEVICE TEST_CONSTEXPR_CXX14 void
test_int_array()
{
    int a[4] = {};
    assert(cuda::std::fill_n(a, UDI(4), static_cast<char>(1)) == a + 4);
    assert(a[0] == 1);
    assert(a[1] == 1);
    assert(a[2] == 1);
    assert(a[3] == 1);
}

// nvcc does not support modification of mutable variables in a constexpr context
struct source {
    TEST_HOST_DEVICE source() : i(0) { }

    TEST_HOST_DEVICE operator int() const { return i++; }
    mutable int i;
};

TEST_HOST_DEVICE void
test_int_array_struct_source()
{
    int a[4] = {};
    assert(cuda::std::fill_n(a, UDI(4), source()) == a + 4);
    assert(a[0] == 0);
    assert(a[1] == 1);
    assert(a[2] == 2);
    assert(a[3] == 3);
}

struct test1 {
    TEST_HOST_DEVICE constexpr test1() : c(0) { }
    TEST_HOST_DEVICE constexpr test1(char xc) : c(xc + 1) { }
    char c;
};

TEST_HOST_DEVICE TEST_CONSTEXPR_CXX14 void
test_struct_array()
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
    TEST_HOST_DEVICE constexpr A() : a_() {}
    TEST_HOST_DEVICE explicit constexpr A(char a) : a_(a) {}
    TEST_HOST_DEVICE constexpr operator unsigned char() const {return 'b';}

    TEST_HOST_DEVICE constexpr friend bool operator==(const A& x, const A& y) {return x.a_ == y.a_;}
};

TEST_HOST_DEVICE TEST_CONSTEXPR_CXX14 void
test5()
{
    A a[3];
    assert(cuda::std::fill_n(&a[0], UDI(3), A('a')) == a+3);
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

TEST_HOST_DEVICE TEST_CONSTEXPR_CXX14 void test6()
{
  Storage foo[5] = {};
  cuda::std::fill_n(&foo[0], UDI(5), Storage());
}

TEST_HOST_DEVICE TEST_CONSTEXPR_CXX14 bool test() {
    test_char<cpp17_output_iterator<char*> >();
    test_char<forward_iterator<char*> >();
    test_char<bidirectional_iterator<char*> >();
    test_char<random_access_iterator<char*> >();
    test_char<char*>();

    test_int<cpp17_output_iterator<int*> >();
    test_int<forward_iterator<int*> >();
    test_int<bidirectional_iterator<int*> >();
    test_int<random_access_iterator<int*> >();
    test_int<int*>();

    test_int_array();
    test_struct_array();

    test5();
    test6();

    return true;
}

int main(int, char**)
{
  test();
  test_int_array_struct_source();
#if TEST_STD_VER >= 2014
  static_assert(test(), "");
#endif // TEST_STD_VER >= 2014

  return 0;
}
