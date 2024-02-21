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

// template<ForwardIterator Iter>
//   requires LessThanComparable<Iter::value_type>
//   Iter
//   max_element(Iter first, Iter last);

#include <cuda/std/__algorithm>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "test_iterators.h"
#include "cases.h"

template <class Iter>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test() {
  Iter first{cuda::std::begin(input_data)};
  Iter last{cuda::std::end(input_data)};

  Iter i = cuda::std::max_element(first, last);
  if (first != last) {
    for (Iter j = first; j != last; ++j)
      assert(!(*i < *j));
  } else
    assert(i == last);
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test() {
  test<forward_iterator<const int*> >();
  test<bidirectional_iterator<const int*> >();
  test<random_access_iterator<const int*> >();
  test<const int*>();

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 2014
  static_assert(test(), "");
#endif

  return 0;
}
