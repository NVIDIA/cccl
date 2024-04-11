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
// cuda::std::pointer_traits specialization

#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/iterator>
#include <cuda/std/type_traits>

#include "test_iterators.h"
#include "test_macros.h"

template <class Iter>
__host__ __device__ TEST_CONSTEXPR_CXX14 bool tests()
{
  using BoundedIter       = cuda::std::__bounded_iter<Iter>;
  using PointerTraits     = cuda::std::pointer_traits<BoundedIter>;
  using BasePointerTraits = cuda::std::pointer_traits<Iter>;
  static_assert(cuda::std::is_same<typename PointerTraits::pointer, BoundedIter>::value, "");
  static_assert(
    cuda::std::is_same<typename PointerTraits::element_type, typename BasePointerTraits::element_type>::value, "");
  static_assert(
    cuda::std::is_same<typename PointerTraits::difference_type, typename BasePointerTraits::difference_type>::value,
    "");

  {
    int array[]                                 = {0, 1, 2, 3, 4};
    int* b                                      = array + 0;
    int* e                                      = array + 5;
    cuda::std::__bounded_iter<Iter> const iter1 = cuda::std::__make_bounded_iter(Iter(b), Iter(b), Iter(e));
    cuda::std::__bounded_iter<Iter> const iter2 = cuda::std::__make_bounded_iter(Iter(e), Iter(b), Iter(e));
    assert(cuda::std::__to_address(iter1) == b); // in-bounds iterator
    assert(cuda::std::__to_address(iter2) == e); // out-of-bounds iterator
#if TEST_STD_VER > 2017
    assert(cuda::std::to_address(iter1) == b); // in-bounds iterator
    assert(cuda::std::to_address(iter2) == e); // out-of-bounds iterator
#endif
  }

  return true;
}

int main(int, char**)
{
  tests<int*>();
#if TEST_STD_VER > 2011
  static_assert(tests<int*>(), "");
#endif

#if TEST_STD_VER > 2017
  tests<contiguous_iterator<int*>>();
  static_assert(tests<contiguous_iterator<int*>>(), "");
#endif

  return 0;
}
