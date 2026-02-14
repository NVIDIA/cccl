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
  {
    using baseIter   = random_access_iterator<int*>;
    using Iter       = cuda::permutation_iterator<baseIter, baseIter>;
    using IterTraits = cuda::std::iterator_traits<Iter>;

    static_assert(cuda::std::same_as<IterTraits::iterator_category, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<IterTraits::value_type, int>);
    static_assert(cuda::std::same_as<IterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::same_as<IterTraits::pointer, void>);
    static_assert(cuda::std::same_as<IterTraits::reference, int&>);
    static_assert(cuda::std::__has_random_access_traversal<Iter>);
  }
  { // still random access
    using baseIter   = contiguous_iterator<int*>;
    using Iter       = cuda::permutation_iterator<baseIter, baseIter>;
    using IterTraits = cuda::std::iterator_traits<Iter>;

    static_assert(cuda::std::same_as<IterTraits::iterator_category, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<IterTraits::value_type, int>);
    static_assert(cuda::std::same_as<IterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::same_as<IterTraits::pointer, void>);
    static_assert(cuda::std::same_as<IterTraits::reference, int&>);
    static_assert(cuda::std::__has_random_access_traversal<Iter>);
  }
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
