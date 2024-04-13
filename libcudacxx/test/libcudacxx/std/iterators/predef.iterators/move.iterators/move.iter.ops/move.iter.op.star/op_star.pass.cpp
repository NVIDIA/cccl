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

// reference operator*() const;
//
//  constexpr in C++17

#include <cuda/std/cassert>
#include <cuda/std/iterator>
#if defined(_LIBCUDACXX_HAS_MEMORY)
#  include <cuda/std/memory>
#endif

#include "test_iterators.h"
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
  cuda::std::move_iterator<It> r(i);
  assert(*r == x);
  typename cuda::std::iterator_traits<It>::value_type x2 = *r;
  assert(x2 == x);
}

struct do_nothing
{
  __host__ __device__ void operator()(void*) const {}
};

int main(int, char**)
{
  {
    A a;
    test(&a, A());
  }
#if defined(_LIBCUDACXX_HAS_MEMORY)
  {
    int i;
    cuda::std::unique_ptr<int, do_nothing> p(&i);
    test(&p, cuda::std::unique_ptr<int, do_nothing>(&i));
  }
#endif
#if TEST_STD_VER > 2011
  {
    constexpr const char* p = "123456789";
    typedef cuda::std::move_iterator<const char*> MI;
    constexpr MI it1 = cuda::std::make_move_iterator(p);
    constexpr MI it2 = cuda::std::make_move_iterator(p + 1);
    static_assert(*it1 == p[0], "");
    static_assert(*it2 == p[1], "");
  }
#endif

#if TEST_STD_VER > 2014
  // Ensure the `iter_move` customization point is being used.
  {
    int a[] = {0, 1, 2};

    int iter_moves  = 0;
    adl::Iterator i = adl::Iterator::TrackMoves(a, iter_moves);
    cuda::std::move_iterator<adl::Iterator> mi(i);

    auto x = *mi;
    assert(x == 0);
    assert(iter_moves == 1);
  }
#endif

  return 0;
}
