//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<InputIterator Iter1, InputIterator Iter2>
//   requires HasEqualTo<Iter1::value_type, Iter2::value_type>
//   constexpr bool     // constexpr after c++17
//   equal(Iter1 first1, Iter1 last1, Iter2 first2);
//
// Introduced in C++14:
// template<InputIterator Iter1, InputIterator Iter2>
//   constexpr bool     // constexpr after c++17
//   equal(Iter1 first1, Iter1 last1, Iter2 first2, Iter2 last2);

#include <cuda/std/algorithm>
#include <cuda/std/cassert>
#include <cuda/std/functional>

#include "test_iterators.h"
#include "test_macros.h"
#include "type_algorithms.h"

TEST_DIAG_SUPPRESS_MSVC(4244) // conversion possible loss of data
TEST_DIAG_SUPPRESS_MSVC(4310) // cast truncates constant value

template <class UnderlyingType, class Iter1>
struct Test
{
  template <class Iter2>
  __host__ __device__ constexpr void operator()()
  {
    UnderlyingType a[]  = {0, 1, 2, 3, 4, 5};
    const unsigned s    = sizeof(a) / sizeof(a[0]);
    UnderlyingType b[s] = {0, 1, 2, 5, 4, 5};

    assert(cuda::std::equal(Iter1(a), Iter1(a + s), Iter2(a)));
    assert(!cuda::std::equal(Iter1(a), Iter1(a + s), Iter2(b)));

    assert(cuda::std::equal(Iter1(a), Iter1(a + s), Iter2(a), cuda::std::equal_to<>()));
    assert(!cuda::std::equal(Iter1(a), Iter1(a + s), Iter2(b), cuda::std::equal_to<>()));

    assert(cuda::std::equal(Iter1(a), Iter1(a + s), Iter2(a), Iter2(a + s)));
    assert(!cuda::std::equal(Iter1(a), Iter1(a + s), Iter2(a), Iter2(a + s - 1)));
    assert(!cuda::std::equal(Iter1(a), Iter1(a + s), Iter2(b), Iter2(b + s)));

    assert(cuda::std::equal(Iter1(a), Iter1(a + s), Iter2(a), Iter2(a + s), cuda::std::equal_to<>()));
    assert(!cuda::std::equal(Iter1(a), Iter1(a + s), Iter2(a), Iter2(a + s - 1), cuda::std::equal_to<>()));
    assert(!cuda::std::equal(Iter1(a), Iter1(a + s), Iter2(b), Iter2(b + s), cuda::std::equal_to<>()));
  }
};

struct TestNarrowingEqualTo
{
  template <class UnderlyingType>
  __host__ __device__ constexpr void operator()()
  {
    UnderlyingType a[] = {
      UnderlyingType(0x1000),
      UnderlyingType(0x1001),
      UnderlyingType(0x1002),
      UnderlyingType(0x1003),
      UnderlyingType(0x1004)};
    UnderlyingType b[] = {
      UnderlyingType(0x1600),
      UnderlyingType(0x1601),
      UnderlyingType(0x1602),
      UnderlyingType(0x1603),
      UnderlyingType(0x1604)};

    assert(cuda::std::equal(a, a + 5, b, cuda::std::equal_to<char>()));
    assert(cuda::std::equal(a, a + 5, b, b + 5, cuda::std::equal_to<char>()));
  }
};

template <class UnderlyingType, class TypeList>
struct TestIter2
{
  template <class Iter1>
  __host__ __device__ constexpr void operator()()
  {
    types::for_each(TypeList(), Test<UnderlyingType, Iter1>());
  }
};

struct AddressCompare
{
  int i = 0;
  __host__ __device__ constexpr AddressCompare(int) {}

  __host__ __device__ operator char()
  {
    return static_cast<char>(i);
  }

  __host__ __device__ friend constexpr bool operator==(const AddressCompare& lhs, const AddressCompare& rhs)
  {
    return &lhs == &rhs;
  }

  __host__ __device__ friend constexpr bool operator!=(const AddressCompare& lhs, const AddressCompare& rhs)
  {
    return &lhs != &rhs;
  }
};

class trivially_equality_comparable
{
public:
  __host__ __device__ constexpr trivially_equality_comparable(int i)
      : i_(i)
  {}
  __host__ __device__ constexpr bool operator==(const trivially_equality_comparable& other) const noexcept
  {
    return i_ == other.i_;
  };

private:
  int i_;
};

__host__ __device__ constexpr bool test()
{
  types::for_each(types::cpp17_input_iterator_list<int*>(), TestIter2<int, types::cpp17_input_iterator_list<int*>>());
  types::for_each(types::cpp17_input_iterator_list<char*>(),
                  TestIter2<char, types::cpp17_input_iterator_list<char*>>());
  types::for_each(types::cpp17_input_iterator_list<AddressCompare*>(),
                  TestIter2<AddressCompare, types::cpp17_input_iterator_list<AddressCompare*>>());

  types::for_each(types::integral_types(), TestNarrowingEqualTo());

  types::for_each(
    types::cpp17_input_iterator_list<trivially_equality_comparable*>{},
    TestIter2<trivially_equality_comparable, types::cpp17_input_iterator_list<trivially_equality_comparable*>>{});

  return true;
}

struct Base
{};
struct Derived : virtual Base
{};

int main(int, char**)
{
  test();
  static_assert(test(), "");

  types::for_each(types::as_pointers<types::cv_qualified_versions<int>>(),
                  TestIter2<int, types::as_pointers<types::cv_qualified_versions<int>>>());
  types::for_each(types::as_pointers<types::cv_qualified_versions<char>>(),
                  TestIter2<char, types::as_pointers<types::cv_qualified_versions<char>>>());

  {
    Derived d;
    Derived* a[] = {&d, nullptr};
    Base* b[]    = {&d, nullptr};

    assert(cuda::std::equal(a, a + 2, b));
    assert(cuda::std::equal(a, a + 2, b, b + 2));
  }

  return 0;
}
