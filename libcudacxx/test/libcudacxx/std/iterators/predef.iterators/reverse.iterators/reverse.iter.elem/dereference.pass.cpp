//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// reverse_iterator

// reference operator*() const; // constexpr in C++17

// Be sure to respect LWG 198:
//    http://www.open-std.org/jtc1/sc22/wg21/docs/lwg-defects.html#198
// LWG 198 was superseded by LWG 2360
//    http://www.open-std.org/jtc1/sc22/wg21/docs/lwg-defects.html#2360

#include <cuda/std/cassert>
#include <cuda/std/iterator>

#include "test_macros.h"

class A
{
  int data_;

public:
  __host__ __device__ A()
      : data_(1)
  {}
  __host__ __device__ ~A()
  {
    data_ = -1;
  }

  __host__ __device__ friend bool operator==(const A& x, const A& y)
  {
    return x.data_ == y.data_;
  }
};

template <class It>
__host__ __device__ void test(It i, typename cuda::std::iterator_traits<It>::value_type x)
{
  cuda::std::reverse_iterator<It> r(i);
  assert(*r == x);
}

int main(int, char**)
{
  A a;
  test(&a + 1, A());

#if TEST_STD_VER > 2011
  {
    constexpr const char* p = "123456789";
    typedef cuda::std::reverse_iterator<const char*> RI;
    constexpr RI it1 = cuda::std::make_reverse_iterator(p + 1);
    constexpr RI it2 = cuda::std::make_reverse_iterator(p + 2);
    static_assert(*it1 == p[0], "");
    static_assert(*it2 == p[1], "");
  }
#endif

  return 0;
}
