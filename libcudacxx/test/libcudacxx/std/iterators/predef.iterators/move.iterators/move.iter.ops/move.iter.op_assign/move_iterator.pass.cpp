//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// move_iterator

// template <class U>
//   requires HasAssign<Iter, const U&>
//   move_iterator&
//   operator=(const move_iterator<U>& u);
//
//  constexpr in C++17

#include <cuda/std/cassert>
#include <cuda/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

template <class It, class U>
__host__ __device__ void test(U u)
{
  const cuda::std::move_iterator<U> r2(u);
  cuda::std::move_iterator<It> r1(It(nullptr));
  cuda::std::move_iterator<It>& rr = (r1 = r2);
  assert(base(r1.base()) == base(u));
  assert(&rr == &r1);
}

struct Base
{};
struct Derived : Base
{};

struct ToIter
{
  typedef cuda::std::forward_iterator_tag iterator_category;
  typedef char* pointer;
  typedef char& reference;
  typedef char value_type;
  typedef signed char difference_type;

  __host__ __device__ explicit TEST_CONSTEXPR_CXX14 ToIter()
      : m_value(0)
  {}
  __host__ __device__ TEST_CONSTEXPR_CXX14 ToIter(const ToIter& src)
      : m_value(src.m_value)
  {}
  // Intentionally not defined, must not be called.
  __host__ __device__ ToIter(char* src);
  __host__ __device__ TEST_CONSTEXPR_CXX14 ToIter& operator=(char* src)
  {
    m_value = src;
    return *this;
  }
  __host__ __device__ TEST_CONSTEXPR_CXX14 ToIter& operator=(const ToIter& src)
  {
    m_value = src.m_value;
    return *this;
  }
  char* m_value;

  __host__ __device__ reference operator*() const;
};

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test_conv_assign()
{
  char c   = '\0';
  char* fi = &c;
  const cuda::std::move_iterator<char*> move_fi(fi);
  cuda::std::move_iterator<ToIter> move_ti;
  move_ti = move_fi;
  assert(move_ti.base().m_value == fi);
  return true;
}

int main(int, char**)
{
  Derived d;

  test<cpp17_input_iterator<Base*>>(cpp17_input_iterator<Derived*>(&d));
  test<forward_iterator<Base*>>(forward_iterator<Derived*>(&d));
  test<bidirectional_iterator<Base*>>(bidirectional_iterator<Derived*>(&d));
  test<random_access_iterator<const Base*>>(random_access_iterator<Derived*>(&d));
  test<Base*>(&d);
  test_conv_assign();
#if TEST_STD_VER > 2011
  {
    using BaseIter             = cuda::std::move_iterator<const Base*>;
    using DerivedIter          = cuda::std::move_iterator<const Derived*>;
    constexpr const Derived* p = nullptr;
    constexpr DerivedIter it1  = cuda::std::make_move_iterator(p);
    constexpr BaseIter it2     = (BaseIter{nullptr} = it1);
    static_assert(it2.base() == p, "");
    static_assert(test_conv_assign(), "");
  }
#endif

  return 0;
}
