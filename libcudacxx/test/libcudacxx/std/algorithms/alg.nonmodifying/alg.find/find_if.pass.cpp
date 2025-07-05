//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<InputIterator Iter, class T>
//   requires HasEqualTo<Iter::value_type, T>
//   constexpr Iter   // constexpr after C++17
//   find_if(Iter first, Iter last, const T& value);

#include <cuda/std/algorithm>
#include <cuda/std/cassert>

#include "test_iterators.h"
#include "test_macros.h"

struct Comparable
{
  int val_;
  __host__ __device__ constexpr Comparable(const int val) noexcept
      : val_(val)
  {}
  __host__ __device__ constexpr bool operator()(const int other) const noexcept
  {
    return val_ == other;
  }
};

__host__ __device__ constexpr bool test()
{
  constexpr int arr[] = {2, 4, 6, 8};
  using Iter          = cpp17_input_iterator<const int*>;

  { // first element matches
    Iter iter = cuda::std::find_if(Iter(cuda::std::begin(arr)), Iter(cuda::std::end(arr)), Comparable(2));
    assert(*iter == 2);
    assert(base(iter) == arr);
  }

  { // range is empty; return last
    Iter iter = cuda::std::find_if(Iter(cuda::std::begin(arr)), Iter(cuda::std::begin(arr)), Comparable(2));
    assert(base(iter) == arr);
  }

  { // if multiple elements match, return the first match
    constexpr int arr_multiple[] = {2, 4, 4, 8};
    Iter iter =
      cuda::std::find_if(Iter(cuda::std::begin(arr_multiple)), Iter(cuda::std::end(arr_multiple)), Comparable(4));
    assert(*iter == 4);
    assert(base(iter) == arr_multiple + 1);
  }

  { // some element matches
    Iter iter = cuda::std::find_if(Iter(cuda::std::begin(arr)), Iter(cuda::std::end(arr)), Comparable(6));
    assert(*iter == 6);
    assert(base(iter) == arr + 2);
  }

  { // last element matches
    Iter iter = cuda::std::find_if(Iter(cuda::std::begin(arr)), Iter(cuda::std::end(arr)), Comparable(8));
    assert(*iter == 8);
    assert(base(iter) == arr + 3);
  }

  { // if no element matches, last is returned
    Iter iter = cuda::std::find_if(Iter(cuda::std::begin(arr)), Iter(cuda::std::end(arr)), Comparable(10));
    assert(base(iter) == arr + 4);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
