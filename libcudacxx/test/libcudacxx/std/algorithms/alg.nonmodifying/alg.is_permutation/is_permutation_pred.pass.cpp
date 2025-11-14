//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<class ForwardIterator1, class ForwardIterator2, class BinaryPredicate>
//   constexpr bool   // constexpr after C++17
//   is_permutation(ForwardIterator1 first1, ForwardIterator1 last1,
//                  ForwardIterator2 first2, BinaryPredicate pred);

#include <cuda/std/algorithm>
#include <cuda/std/cassert>
#include <cuda/std/functional>

#include "counting_predicates.h"
#include "test_iterators.h"
#include "test_macros.h"

struct S
{
  __host__ __device__ constexpr S(int i)
      : i_(i)
  {}
  __host__ __device__ constexpr bool operator==(const S& other) = delete;
  int i_;
};

struct eq
{
  __host__ __device__ constexpr bool operator()(const S& a, const S& b)
  {
    return a.i_ == b.i_;
  }
};

__host__ __device__ constexpr bool test()
{
  {
    const int ia[]    = {0};
    const int ib[]    = {0};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + 0),
             forward_iterator<const int*>(ib),
             cuda::std::equal_to<const int>())
           == true);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             cuda::std::equal_to<const int>())
           == true);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa),
             cuda::std::equal_to<const int>())
           == true);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa - 1),
             cuda::std::equal_to<const int>())
           == false);
  }
  {
    const int ia[]    = {0};
    const int ib[]    = {1};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             cuda::std::equal_to<const int>())
           == false);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa),
             cuda::std::equal_to<const int>())
           == false);
  }

  {
    const int ia[]    = {0, 0};
    const int ib[]    = {0, 0};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             cuda::std::equal_to<const int>())
           == true);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa),
             cuda::std::equal_to<const int>())
           == true);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa - 1),
             cuda::std::equal_to<const int>())
           == false);
  }
  {
    const int ia[]    = {0, 0};
    const int ib[]    = {0, 1};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             cuda::std::equal_to<const int>())
           == false);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa),
             cuda::std::equal_to<const int>())
           == false);
  }
  {
    const int ia[]    = {0, 0};
    const int ib[]    = {1, 0};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             cuda::std::equal_to<const int>())
           == false);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa),
             cuda::std::equal_to<const int>())
           == false);
  }
  {
    const int ia[]    = {0, 0};
    const int ib[]    = {1, 1};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             cuda::std::equal_to<const int>())
           == false);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa),
             cuda::std::equal_to<const int>())
           == false);
  }
  {
    const int ia[]    = {0, 1};
    const int ib[]    = {0, 0};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             cuda::std::equal_to<const int>())
           == false);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa),
             cuda::std::equal_to<const int>())
           == false);
  }
  {
    const int ia[]    = {0, 1};
    const int ib[]    = {0, 1};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             cuda::std::equal_to<const int>())
           == true);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa),
             cuda::std::equal_to<const int>())
           == true);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa - 1),
             cuda::std::equal_to<const int>())
           == false);
  }
  {
    const int ia[]    = {0, 1};
    const int ib[]    = {1, 0};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             cuda::std::equal_to<const int>())
           == true);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa),
             cuda::std::equal_to<const int>())
           == true);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa - 1),
             cuda::std::equal_to<const int>())
           == false);
  }
  {
    const int ia[]    = {0, 1};
    const int ib[]    = {1, 1};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             cuda::std::equal_to<const int>())
           == false);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa),
             cuda::std::equal_to<const int>())
           == false);
  }
  {
    const int ia[]    = {1, 0};
    const int ib[]    = {0, 0};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             cuda::std::equal_to<const int>())
           == false);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa),
             cuda::std::equal_to<const int>())
           == false);
  }
  {
    const int ia[]    = {1, 0};
    const int ib[]    = {0, 1};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             cuda::std::equal_to<const int>())
           == true);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa),
             cuda::std::equal_to<const int>())
           == true);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa - 1),
             cuda::std::equal_to<const int>())
           == false);
  }
  {
    const int ia[]    = {1, 0};
    const int ib[]    = {1, 0};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             cuda::std::equal_to<const int>())
           == true);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa),
             cuda::std::equal_to<const int>())
           == true);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa - 1),
             cuda::std::equal_to<const int>())
           == false);
  }
  {
    const int ia[]    = {1, 0};
    const int ib[]    = {1, 1};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             cuda::std::equal_to<const int>())
           == false);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa),
             cuda::std::equal_to<const int>())
           == false);
  }
  {
    const int ia[]    = {1, 1};
    const int ib[]    = {0, 0};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             cuda::std::equal_to<const int>())
           == false);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa),
             cuda::std::equal_to<const int>())
           == false);
  }
  {
    const int ia[]    = {1, 1};
    const int ib[]    = {0, 1};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             cuda::std::equal_to<const int>())
           == false);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa),
             cuda::std::equal_to<const int>())
           == false);
  }
  {
    const int ia[]    = {1, 1};
    const int ib[]    = {1, 0};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             cuda::std::equal_to<const int>())
           == false);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa),
             cuda::std::equal_to<const int>())
           == false);
  }
  {
    const int ia[]    = {1, 1};
    const int ib[]    = {1, 1};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             cuda::std::equal_to<const int>())
           == true);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa),
             cuda::std::equal_to<const int>())
           == true);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa - 1),
             cuda::std::equal_to<const int>())
           == false);
  }

  {
    const int ia[]    = {0, 0, 0};
    const int ib[]    = {1, 0, 0};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             cuda::std::equal_to<const int>())
           == false);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa),
             cuda::std::equal_to<const int>())
           == false);
  }
  {
    const int ia[]    = {0, 0, 0};
    const int ib[]    = {1, 0, 1};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             cuda::std::equal_to<const int>())
           == false);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa),
             cuda::std::equal_to<const int>())
           == false);
  }
  {
    const int ia[]    = {0, 0, 0};
    const int ib[]    = {1, 0, 2};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             cuda::std::equal_to<const int>())
           == false);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa),
             cuda::std::equal_to<const int>())
           == false);
  }
  {
    const int ia[]    = {0, 0, 0};
    const int ib[]    = {1, 1, 0};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             cuda::std::equal_to<const int>())
           == false);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa),
             cuda::std::equal_to<const int>())
           == false);
  }
  {
    const int ia[]    = {0, 0, 0};
    const int ib[]    = {1, 1, 1};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             cuda::std::equal_to<const int>())
           == false);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa),
             cuda::std::equal_to<const int>())
           == false);
  }
  {
    const int ia[]    = {0, 0, 0};
    const int ib[]    = {1, 1, 2};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             cuda::std::equal_to<const int>())
           == false);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa),
             cuda::std::equal_to<const int>())
           == false);
  }
  {
    const int ia[]    = {0, 0, 0};
    const int ib[]    = {1, 2, 0};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             cuda::std::equal_to<const int>())
           == false);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa),
             cuda::std::equal_to<const int>())
           == false);
  }
  {
    const int ia[]    = {0, 0, 0};
    const int ib[]    = {1, 2, 1};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             cuda::std::equal_to<const int>())
           == false);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa),
             cuda::std::equal_to<const int>())
           == false);
  }
  {
    const int ia[]    = {0, 0, 0};
    const int ib[]    = {1, 2, 2};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             cuda::std::equal_to<const int>())
           == false);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa),
             cuda::std::equal_to<const int>())
           == false);
  }
  {
    const int ia[]    = {0, 0, 1};
    const int ib[]    = {1, 0, 0};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             cuda::std::equal_to<const int>())
           == true);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa),
             cuda::std::equal_to<const int>())
           == true);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa - 1),
             cuda::std::equal_to<const int>())
           == false);
  }
  {
    const int ia[]    = {0, 0, 1};
    const int ib[]    = {1, 0, 1};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             cuda::std::equal_to<const int>())
           == false);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa),
             cuda::std::equal_to<const int>())
           == false);
  }
  {
    const int ia[]    = {0, 1, 2};
    const int ib[]    = {1, 0, 2};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             cuda::std::equal_to<const int>())
           == true);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa),
             cuda::std::equal_to<const int>())
           == true);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa - 1),
             cuda::std::equal_to<const int>())
           == false);
  }
  {
    const int ia[]    = {0, 1, 2};
    const int ib[]    = {1, 2, 0};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             cuda::std::equal_to<const int>())
           == true);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa),
             cuda::std::equal_to<const int>())
           == true);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa - 1),
             cuda::std::equal_to<const int>())
           == false);
  }
  {
    const int ia[]    = {0, 1, 2};
    const int ib[]    = {2, 1, 0};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             cuda::std::equal_to<const int>())
           == true);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa),
             cuda::std::equal_to<const int>())
           == true);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa - 1),
             cuda::std::equal_to<const int>())
           == false);
  }
  {
    const int ia[]    = {0, 1, 2};
    const int ib[]    = {2, 0, 1};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             cuda::std::equal_to<const int>())
           == true);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa),
             cuda::std::equal_to<const int>())
           == true);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa - 1),
             cuda::std::equal_to<const int>())
           == false);
  }
  {
    const int ia[]    = {0, 0, 1};
    const int ib[]    = {1, 0, 1};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             cuda::std::equal_to<const int>())
           == false);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa),
             cuda::std::equal_to<const int>())
           == false);
  }
  {
    const int ia[]    = {0, 0, 1};
    const int ib[]    = {1, 0, 0};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             cuda::std::equal_to<const int>())
           == true);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa),
             cuda::std::equal_to<const int>())
           == true);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib + 1),
             forward_iterator<const int*>(ib + sa),
             cuda::std::equal_to<const int>())
           == false);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa - 1),
             cuda::std::equal_to<const int>())
           == false);
  }
  {
    const int ia[]    = {0, 1, 2, 3, 0, 5, 6, 2, 4, 4};
    const int ib[]    = {4, 2, 3, 0, 1, 4, 0, 5, 6, 2};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             cuda::std::equal_to<const int>())
           == true);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa),
             cuda::std::equal_to<const int>())
           == true);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib + 1),
             forward_iterator<const int*>(ib + sa),
             cuda::std::equal_to<const int>())
           == false);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa - 1),
             cuda::std::equal_to<const int>())
           == false);

    int comparison_count = 0;
    counting_predicate<cuda::std::equal_to<const int>> counting_equals(
      cuda::std::equal_to<const int>(), comparison_count);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa - 1),
             counting_equals)
           == false);
    assert(comparison_count > 0);
    comparison_count = 0;
    assert(cuda::std::is_permutation(
             random_access_iterator<const int*>(ia),
             random_access_iterator<const int*>(ia + sa),
             random_access_iterator<const int*>(ib),
             random_access_iterator<const int*>(ib + sa - 1),
             counting_equals)
           == false);
    assert(comparison_count == 0);
  }
  {
    const int ia[]    = {0, 1, 2, 3, 0, 5, 6, 2, 4, 4};
    const int ib[]    = {4, 2, 3, 0, 1, 4, 0, 5, 6, 0};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             cuda::std::equal_to<const int>())
           == false);

    assert(cuda::std::is_permutation(
             forward_iterator<const int*>(ia),
             forward_iterator<const int*>(ia + sa),
             forward_iterator<const int*>(ib),
             forward_iterator<const int*>(ib + sa),
             cuda::std::equal_to<const int>())
           == false);
  }
  {
    const S a[]       = {S(0), S(1)};
    const S b[]       = {S(1), S(0)};
    const unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(cuda::std::is_permutation(
      forward_iterator<const S*>(a), forward_iterator<const S*>(a + sa), forward_iterator<const S*>(b), eq()));

    assert(cuda::std::is_permutation(
      forward_iterator<const S*>(a),
      forward_iterator<const S*>(a + sa),
      forward_iterator<const S*>(b),
      forward_iterator<const S*>(b + sa),
      eq()));
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
