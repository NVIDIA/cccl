//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/iterator>

#include "test_iterators.h"
#include "test_macros.h"

#if !TEST_COMPILER(NVRTC)
#  include <iterator>
#endif // !TEST_COMPILER(NVRTC)

template <template <class...> class Traits>
__host__ __device__ void test()
{
  using Iter       = cuda::discard_iterator;
  using IterTraits = Traits<Iter>;

  static_assert(cuda::std::same_as<typename IterTraits::iterator_category, cuda::std::random_access_iterator_tag>);
  static_assert(cuda::std::same_as<typename IterTraits::value_type, void>);
  static_assert(cuda::std::same_as<typename IterTraits::difference_type, cuda::std::ptrdiff_t>);
  static_assert(cuda::std::same_as<typename IterTraits::pointer, void>);
  static_assert(cuda::std::same_as<typename IterTraits::reference, void>);
  static_assert(cuda::std::input_or_output_iterator<cuda::discard_iterator>);
  static_assert(cuda::std::output_iterator<cuda::discard_iterator, float>);
  static_assert(cuda::std::__has_random_access_traversal<Iter>);
}

__host__ __device__ void test()
{
  test<cuda::std::iterator_traits>();
#if !TEST_COMPILER(NVRTC)
  test<std::iterator_traits>();
#endif // !TEST_COMPILER(NVRTC)
}

int main(int, char**)
{
  return 0;
}
