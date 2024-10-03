//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template <class _Iterator>
// struct __bounded_iter;
//
// Dereference and indexing operators

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03
// ADDITIONAL_COMPILE_FLAGS: -D_CCCL_ENABLE_ASSERTIONS

#include <cuda/std/iterator>

#include "check_assertion.h"
#include "test_iterators.h"
#include "test_macros.h"

struct Foo
{
  int x;
  __host__ __device__ TEST_CONSTEXPR bool operator==(Foo const& other) const
  {
    return x == other.x;
  }
};

template <class Iter>
__host__ __device__ TEST_CONSTEXPR_CXX14 bool tests()
{
  Foo array[]                                 = {Foo{40}, Foo{41}, Foo{42}, Foo{43}, Foo{44}};
  Foo* b                                      = array + 0;
  Foo* e                                      = array + 5;
  cuda::std::__bounded_iter<Iter> const iter1 = cuda::std::__make_bounded_iter(Iter(b), Iter(b), Iter(e));
  cuda::std::__bounded_iter<Iter> const iter2 = cuda::std::__make_bounded_iter(Iter(e), Iter(b), Iter(e));

  // operator*
  assert(*iter1 == Foo{40});
  // operator->
  assert(iter1->x == 40);
  // operator[]
  assert(iter1[0] == Foo{40});
  assert(iter1[1] == Foo{41});
  assert(iter1[2] == Foo{42});
  assert(iter2[-1] == Foo{44});
  assert(iter2[-2] == Foo{43});

  return true;
}

template <class Iter>
__host__ __device__ void test_death()
{
  Foo array[]                                = {Foo{0}, Foo{1}, Foo{2}, Foo{3}, Foo{4}};
  Foo* b                                     = array + 0;
  Foo* e                                     = array + 5;
  cuda::std::__bounded_iter<Iter> const iter = cuda::std::__make_bounded_iter(Iter(b), Iter(b), Iter(e));
  cuda::std::__bounded_iter<Iter> const oob  = cuda::std::__make_bounded_iter(Iter(e), Iter(b), Iter(e));

  // operator*
  TEST_CCCL_ASSERT_FAILURE(*oob, "__bounded_iter::operator*: Attempt to dereference an out-of-range iterator");
  // operator->
  TEST_CCCL_ASSERT_FAILURE(oob->x, "__bounded_iter::operator->: Attempt to dereference an out-of-range iterator");
  // operator[]
  TEST_CCCL_ASSERT_FAILURE(iter[-1], "__bounded_iter::operator[]: Attempt to index an iterator out-of-range");
  TEST_CCCL_ASSERT_FAILURE(iter[5], "__bounded_iter::operator[]: Attempt to index an iterator out-of-range");
  TEST_CCCL_ASSERT_FAILURE(oob[0], "__bounded_iter::operator[]: Attempt to index an iterator out-of-range");
  TEST_CCCL_ASSERT_FAILURE(oob[1], "__bounded_iter::operator[]: Attempt to index an iterator out-of-range");
  TEST_CCCL_ASSERT_FAILURE(oob[-6], "__bounded_iter::operator[]: Attempt to index an iterator out-of-range");
}

int main(int, char**)
{
  tests<Foo*>();
  test_death<Foo*>();
#if TEST_STD_VER > 2011
  static_assert(tests<Foo*>(), "");
#endif

#if TEST_STD_VER > 2017
  tests<contiguous_iterator<Foo*>>();
  test_death<contiguous_iterator<Foo*>>();
  static_assert(tests<contiguous_iterator<Foo*>>(), "");
#endif

  return 0;
}
