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

// explicit move_iterator(Iter i);
//
//  constexpr in C++17

#include <cuda/std/cassert>
#include <cuda/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

template <class It>
__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  static_assert(cuda::std::is_constructible<cuda::std::move_iterator<It>, const It&>::value, "");
  static_assert(cuda::std::is_constructible<cuda::std::move_iterator<It>, It&&>::value, "");
  static_assert(!cuda::std::is_convertible<const It&, cuda::std::move_iterator<It>>::value, "");
  static_assert(!cuda::std::is_convertible<It&&, cuda::std::move_iterator<It>>::value, "");

  char s[] = "123";
  {
    It it = It(s);
    cuda::std::move_iterator<It> r(it);
    assert(base(r.base()) == s);
  }
  {
    It it = It(s);
    cuda::std::move_iterator<It> r(cuda::std::move(it));
    assert(base(r.base()) == s);
  }
  return true;
}

template <class It>
__host__ __device__ TEST_CONSTEXPR_CXX14 bool test_moveonly()
{
#if !defined(TEST_COMPILER_MSVC_2017)
  static_assert(!cuda::std::is_constructible<cuda::std::move_iterator<It>, const It&>::value, "");
#endif // !TEST_COMPILER_MSVC_2017
  static_assert(cuda::std::is_constructible<cuda::std::move_iterator<It>, It&&>::value, "");
  static_assert(!cuda::std::is_convertible<const It&, cuda::std::move_iterator<It>>::value, "");
  static_assert(!cuda::std::is_convertible<It&&, cuda::std::move_iterator<It>>::value, "");

  char s[] = "123";
  {
    It it = It(s);
    cuda::std::move_iterator<It> r(cuda::std::move(it));
    assert(base(r.base()) == s);
  }
  return true;
}

int main(int, char**)
{
  test<cpp17_input_iterator<char*>>();
  test<forward_iterator<char*>>();
  test<bidirectional_iterator<char*>>();
  test<random_access_iterator<char*>>();
  test<char*>();
  test<const char*>();

#if TEST_STD_VER > 2011
  static_assert(test<cpp17_input_iterator<char*>>(), "");
  static_assert(test<forward_iterator<char*>>(), "");
  static_assert(test<bidirectional_iterator<char*>>(), "");
  static_assert(test<random_access_iterator<char*>>(), "");
  static_assert(test<char*>(), "");
  static_assert(test<const char*>(), "");
#endif // TEST_STD_VER > 2011

#if TEST_STD_VER > 2014
  test<contiguous_iterator<char*>>();
  test_moveonly<cpp20_input_iterator<char*>>();
  static_assert(test<contiguous_iterator<char*>>(), "");
  static_assert(test_moveonly<cpp20_input_iterator<char*>>(), "");
#endif // TEST_STD_VER > 2014

  return 0;
}
