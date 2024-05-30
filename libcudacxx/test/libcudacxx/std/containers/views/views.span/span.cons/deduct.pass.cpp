//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14

// gcc does not support deduction guides until gcc-7 and that is buggy
// UNSUPPORTED: gcc-6, gcc-7

// <span>

//   template<class It, class EndOrSize>
//     span(It, EndOrSize) -> span<remove_reference_t<iter_reference_t<_It>>>;
//
//   template<class T, size_t N>
//     span(T (&)[N]) -> span<T, N>;
//
//   template<class T, size_t N>
//     span(array<T, N>&) -> span<T, N>;
//
//   template<class T, size_t N>
//     span(const array<T, N>&) -> span<const T, N>;
//
//   template<class R>
//     span(R&&) -> span<remove_reference_t<ranges::range_reference_t<R>>>;

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/iterator>
#include <cuda/std/span>
#include <cuda/std/type_traits>

#include "test_macros.h"

__host__ __device__ void test_iterator_sentinel()
{
  int arr[] = {1, 2, 3};
  {
    cuda::std::span s{cuda::std::begin(arr), cuda::std::end(arr)};
    ASSERT_SAME_TYPE(decltype(s), cuda::std::span<int>);
    assert(s.size() == cuda::std::size(arr));
    assert(s.data() == cuda::std::data(arr));
  }
  {
    cuda::std::span s{cuda::std::begin(arr), 3};
    ASSERT_SAME_TYPE(decltype(s), cuda::std::span<int>);
    assert(s.size() == cuda::std::size(arr));
    assert(s.data() == cuda::std::data(arr));
  }

#if !defined(TEST_COMPILER_MSVC)
  // P3029R1: deduction from `integral_constant`
  {
    cuda::std::span s{cuda::std::begin(arr), cuda::std::integral_constant<size_t, 3>{}};
    ASSERT_SAME_TYPE(decltype(s), cuda::std::span<int, 3>);
    assert(s.size() == cuda::std::size(arr));
    assert(s.data() == cuda::std::data(arr));
  }
#endif // !TEST_COMPILER_MSVC
}

__host__ __device__ void test_c_array()
{
  {
    int arr[] = {1, 2, 3};
    cuda::std::span s{arr};
    ASSERT_SAME_TYPE(decltype(s), cuda::std::span<int, 3>);
    assert(s.size() == cuda::std::size(arr));
    assert(s.data() == cuda::std::data(arr));
  }

  {
    const int arr[] = {1, 2, 3};
    cuda::std::span s{arr};
    ASSERT_SAME_TYPE(decltype(s), cuda::std::span<const int, 3>);
    assert(s.size() == cuda::std::size(arr));
    assert(s.data() == cuda::std::data(arr));
  }
}

__host__ __device__ void test_std_array()
{
  {
    cuda::std::array<double, 4> arr = {1.0, 2.0, 3.0, 4.0};
    cuda::std::span s{arr};
    ASSERT_SAME_TYPE(decltype(s), cuda::std::span<double, 4>);
    assert(s.size() == arr.size());
    assert(s.data() == arr.data());
  }

  {
    const cuda::std::array<long, 5> arr = {4, 5, 6, 7, 8};
    cuda::std::span s{arr};
    ASSERT_SAME_TYPE(decltype(s), cuda::std::span<const long, 5>);
    assert(s.size() == arr.size());
    assert(s.data() == arr.data());
  }
}

int main(int, char**)
{
  test_iterator_sentinel();
  test_c_array();
  test_std_array();

  return 0;
}
