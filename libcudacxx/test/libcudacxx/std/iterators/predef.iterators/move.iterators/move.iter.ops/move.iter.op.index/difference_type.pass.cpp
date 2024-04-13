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

// constexpr reference operator[](difference_type n) const; // Return type unspecified until C++20

#include <cuda/std/cassert>
#include <cuda/std/iterator>
#if defined(_LIBCUDACXX_HAS_MEMORY)
#  include <cuda/std/memory>
#endif

#include "test_iterators.h"
#include "test_macros.h"

#if defined(TEST_COMPILER_MSVC)
#  pragma warning(disable : 4172) // returning address of local variable or temporary
#endif // TEST_COMPILER_MSVC

template <class It>
__host__ __device__ void test(It i,
                              typename cuda::std::iterator_traits<It>::difference_type n,
                              typename cuda::std::iterator_traits<It>::value_type x)
{
  typedef typename cuda::std::iterator_traits<It>::value_type value_type;
  const cuda::std::move_iterator<It> r(i);
  value_type rr = r[n];
  assert(rr == x);
}

struct do_nothing
{
  __host__ __device__ void operator()(void*) const {}
};

int main(int, char**)
{
  {
    char s[] = "1234567890";
#if defined(TEST_COMPILER_NVHPC)
    for (int i = 0; i < 10; ++i)
    {
      s[i] = i == 9 ? '0' : ('1' + i);
    }
#endif // TEST_COMPILER_NVHPC
    test(random_access_iterator<char*>(s + 5), 4, '0');
    test(s + 5, 4, '0');
  }
#if defined(_LIBCUDACXX_HAS_MEMORY)
  {
    int i[5];
    typedef cuda::std::unique_ptr<int, do_nothing> Ptr;
    Ptr p[5];
    for (unsigned j = 0; j < 5; ++j)
    {
      p[j].reset(i + j);
    }
    test(p, 3, Ptr(i + 3));
  }
#endif // _LIBCUDACXX_HAS_MEMORY
#if TEST_STD_VER > 2011 && (!defined(TEST_COMPILER_MSVC) || TEST_STD_VER > 2014) // MSVC bails here
  {
    constexpr const char* p = "123456789";
    typedef cuda::std::move_iterator<const char*> MI;
    constexpr MI it1 = cuda::std::make_move_iterator(p);
    static_assert(it1[0] == '1', "");
    static_assert(it1[5] == '6', "");
  }
#endif // TEST_STD_VER > 2011 && (!defined(TEST_COMPILER_MSVC) || TEST_STD_VER > 2014)

#if TEST_STD_VER > 2014
  // Ensure the `iter_move` customization point is being used.
  {
    int a[] = {0, 1, 2};

    int iter_moves  = 0;
    adl::Iterator i = adl::Iterator::TrackMoves(a, iter_moves);
    cuda::std::move_iterator<adl::Iterator> mi(i);

    auto x = mi[0];
    assert(x == 0);
    assert(iter_moves == 1);
  }
#endif // TEST_STD_VER > 2014

  return 0;
}
