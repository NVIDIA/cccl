//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// cuda::std::ranges::filter_view<V>::<iterator>() requires default_initializable<iterator_t<V>> = default;

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>

#include "../types.h"
#include "test_iterators.h"
#include "test_macros.h"

template <class Iterator, bool IsNoexcept>
__host__ __device__ constexpr void test_default_constructible()
{
  // Make sure the iterator is default constructible when the underlying iterator is.
  using View           = minimal_view<Iterator, sentinel_wrapper<Iterator>>;
  using FilterView     = cuda::std::ranges::filter_view<View, AlwaysTrue>;
  using FilterIterator = cuda::std::ranges::iterator_t<FilterView>;
  FilterIterator iter1{};
  FilterIterator iter2;
  assert(iter1 == iter2);
#if !defined(TEST_COMPILER_NVCC) && !defined(TEST_COMPILER_NVRTC)
  static_assert(noexcept(FilterIterator()) == IsNoexcept);
#endif // !TEST_COMPILER_NVCC && !TEST_COMPILER_NVRTC
}

template <class Iterator>
__host__ __device__ constexpr void test_not_default_constructible()
{
  // Make sure the iterator is *not* default constructible when the underlying iterator isn't.
  using View           = minimal_view<Iterator, sentinel_wrapper<Iterator>>;
  using FilterView     = cuda::std::ranges::filter_view<View, AlwaysTrue>;
  using FilterIterator = cuda::std::ranges::iterator_t<FilterView>;
  static_assert(!cuda::std::is_default_constructible_v<FilterIterator>);
}

__host__ __device__ constexpr bool tests()
{
  test_not_default_constructible<cpp17_input_iterator<int*>>();
  test_not_default_constructible<cpp20_input_iterator<int*>>();
  test_default_constructible<forward_iterator<int*>, /* noexcept */ false>();
  test_default_constructible<bidirectional_iterator<int*>, /* noexcept */ false>();
  test_default_constructible<random_access_iterator<int*>, /* noexcept */ false>();
  test_default_constructible<contiguous_iterator<int*>, /* noexcept */ false>();
  test_default_constructible<int*, /* noexcept */ true>();
  return true;
}

int main(int, char**)
{
  tests();
  static_assert(tests(), "");
  return 0;
}
