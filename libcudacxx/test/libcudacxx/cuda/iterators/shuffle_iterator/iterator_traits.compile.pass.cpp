//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Test iterator category and iterator concepts.

#include <cuda/iterator>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>

#include "test_macros.h"
#include "types.h"

#if !TEST_COMPILER(NVRTC)
#  include <iterator>
#endif // !TEST_COMPILER(NVRTC)

template <template <class...> class Traits>
__host__ __device__ void test()
{
  {
    using Iter       = cuda::shuffle_iterator<char, fake_bijection<>>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::same_as<typename IterTraits::iterator_category, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<typename IterTraits::value_type, char>);
    static_assert(cuda::std::is_same_v<typename IterTraits::difference_type,
                                       cuda::std::make_signed_t<typename IterTraits::value_type>>);
    static_assert(cuda::std::is_signed_v<typename IterTraits::difference_type>);
    static_assert(cuda::std::same_as<typename IterTraits::difference_type, signed char>);
    static_assert(cuda::std::random_access_iterator<Iter>);
    static_assert(cuda::std::__has_random_access_traversal<Iter>);
  }

  {
    using Iter       = cuda::shuffle_iterator<short, fake_bijection<>>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::same_as<typename IterTraits::iterator_category, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<typename IterTraits::value_type, short>);
    static_assert(cuda::std::is_same_v<typename IterTraits::difference_type,
                                       cuda::std::make_signed_t<typename IterTraits::value_type>>);
    static_assert(cuda::std::is_signed_v<typename IterTraits::difference_type>);
    static_assert(cuda::std::same_as<typename IterTraits::difference_type, short>);
    static_assert(cuda::std::random_access_iterator<Iter>);
    static_assert(cuda::std::__has_random_access_traversal<Iter>);
  }

  {
    using Iter       = cuda::shuffle_iterator<size_t, fake_bijection<>>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::same_as<typename IterTraits::iterator_category, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<typename IterTraits::value_type, size_t>);
    static_assert(cuda::std::is_same_v<typename IterTraits::difference_type,
                                       cuda::std::make_signed_t<typename IterTraits::value_type>>);
    static_assert(cuda::std::is_signed_v<typename IterTraits::difference_type>);
    static_assert(cuda::std::same_as<typename IterTraits::difference_type, ptrdiff_t>);
    static_assert(cuda::std::random_access_iterator<Iter>);
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
