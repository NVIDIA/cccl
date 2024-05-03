//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<ForwardIterator Iter, StrictWeakOrder<auto, Iter::value_type> Compare>
//   requires CopyConstructible<Compare>
//   bool
//   is_sorted(Iter first, Iter last, Compare comp);

#include <cuda/std/__algorithm_>
#include <cuda/std/cassert>
#include <cuda/std/functional>

#include "test_iterators.h"
#include "test_macros.h"

template <class Iter>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test()
{
  {
    int a[]     = {0};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(cuda::std::is_sorted(Iter(a), Iter(a)));
    assert(cuda::std::is_sorted(Iter(a), Iter(a + sa), cuda::std::greater<int>()));
  }

  {
    int a[]     = {0, 0};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(cuda::std::is_sorted(Iter(a), Iter(a + sa), cuda::std::greater<int>()));
  }
  {
    int a[]     = {0, 1};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(!cuda::std::is_sorted(Iter(a), Iter(a + sa), cuda::std::greater<int>()));
  }
  {
    int a[]     = {1, 0};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(cuda::std::is_sorted(Iter(a), Iter(a + sa), cuda::std::greater<int>()));
  }
  {
    int a[]     = {1, 1};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(cuda::std::is_sorted(Iter(a), Iter(a + sa), cuda::std::greater<int>()));
  }

  {
    int a[]     = {0, 0, 0};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(cuda::std::is_sorted(Iter(a), Iter(a + sa), cuda::std::greater<int>()));
  }
  {
    int a[]     = {0, 0, 1};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(!cuda::std::is_sorted(Iter(a), Iter(a + sa), cuda::std::greater<int>()));
  }
  {
    int a[]     = {0, 1, 0};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(!cuda::std::is_sorted(Iter(a), Iter(a + sa), cuda::std::greater<int>()));
  }
  {
    int a[]     = {0, 1, 1};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(!cuda::std::is_sorted(Iter(a), Iter(a + sa), cuda::std::greater<int>()));
  }
  {
    int a[]     = {1, 0, 0};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(cuda::std::is_sorted(Iter(a), Iter(a + sa), cuda::std::greater<int>()));
  }
  {
    int a[]     = {1, 0, 1};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(!cuda::std::is_sorted(Iter(a), Iter(a + sa), cuda::std::greater<int>()));
  }
  {
    int a[]     = {1, 1, 0};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(cuda::std::is_sorted(Iter(a), Iter(a + sa), cuda::std::greater<int>()));
  }
  {
    int a[]     = {1, 1, 1};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(cuda::std::is_sorted(Iter(a), Iter(a + sa), cuda::std::greater<int>()));
  }

  {
    int a[]     = {0, 0, 0, 0};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(cuda::std::is_sorted(Iter(a), Iter(a + sa), cuda::std::greater<int>()));
  }
  {
    int a[]     = {0, 0, 0, 1};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(!cuda::std::is_sorted(Iter(a), Iter(a + sa), cuda::std::greater<int>()));
  }
  {
    int a[]     = {0, 0, 1, 0};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(!cuda::std::is_sorted(Iter(a), Iter(a + sa), cuda::std::greater<int>()));
  }
  {
    int a[]     = {0, 0, 1, 1};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(!cuda::std::is_sorted(Iter(a), Iter(a + sa), cuda::std::greater<int>()));
  }
  {
    int a[]     = {0, 1, 0, 0};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(!cuda::std::is_sorted(Iter(a), Iter(a + sa), cuda::std::greater<int>()));
  }
  {
    int a[]     = {0, 1, 0, 1};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(!cuda::std::is_sorted(Iter(a), Iter(a + sa), cuda::std::greater<int>()));
  }
  {
    int a[]     = {0, 1, 1, 0};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(!cuda::std::is_sorted(Iter(a), Iter(a + sa), cuda::std::greater<int>()));
  }
  {
    int a[]     = {0, 1, 1, 1};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(!cuda::std::is_sorted(Iter(a), Iter(a + sa), cuda::std::greater<int>()));
  }
  {
    int a[]     = {1, 0, 0, 0};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(cuda::std::is_sorted(Iter(a), Iter(a + sa), cuda::std::greater<int>()));
  }
  {
    int a[]     = {1, 0, 0, 1};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(!cuda::std::is_sorted(Iter(a), Iter(a + sa), cuda::std::greater<int>()));
  }
  {
    int a[]     = {1, 0, 1, 0};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(!cuda::std::is_sorted(Iter(a), Iter(a + sa), cuda::std::greater<int>()));
  }
  {
    int a[]     = {1, 0, 1, 1};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(!cuda::std::is_sorted(Iter(a), Iter(a + sa), cuda::std::greater<int>()));
  }
  {
    int a[]     = {1, 1, 0, 0};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(cuda::std::is_sorted(Iter(a), Iter(a + sa), cuda::std::greater<int>()));
  }
  {
    int a[]     = {1, 1, 0, 1};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(!cuda::std::is_sorted(Iter(a), Iter(a + sa), cuda::std::greater<int>()));
  }
  {
    int a[]     = {1, 1, 1, 0};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(cuda::std::is_sorted(Iter(a), Iter(a + sa), cuda::std::greater<int>()));
  }
  {
    int a[]     = {1, 1, 1, 1};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(cuda::std::is_sorted(Iter(a), Iter(a + sa), cuda::std::greater<int>()));
  }
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  test<forward_iterator<const int*>>();
  test<bidirectional_iterator<const int*>>();
  test<random_access_iterator<const int*>>();
  test<const int*>();

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2014
  static_assert(test(), "");
#endif

  return 0;
}
