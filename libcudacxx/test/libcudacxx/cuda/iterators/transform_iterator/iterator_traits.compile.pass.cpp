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
#include <cuda/std/cassert>
#include <cuda/std/concepts>

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

template <class Iter, class Fn>
_CCCL_CONCEPT HasIterCategory =
  _CCCL_REQUIRES_EXPR((Iter, Fn))(typename(typename cuda::transform_iterator<Iter, Fn>::iterator_category));

#if !TEST_COMPILER(NVRTC)
#  include <iterator>
#endif // !TEST_COMPILER(NVRTC)

template <template <class...> class Traits>
__host__ __device__ constexpr bool test()
{
  {
    using Iter       = cuda::transform_iterator<int*, Increment>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::same_as<typename IterTraits::iterator_category, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<typename IterTraits::value_type, int>);
    static_assert(cuda::std::same_as<typename IterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::random_access_iterator<Iter>);
    static_assert(cuda::std::__is_cpp17_random_access_iterator<Iter>);
  }
  {
    // Member typedefs for random access iterator.
    using Iter       = cuda::transform_iterator<random_access_iterator<int*>, Increment>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::same_as<typename IterTraits::iterator_category, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<typename IterTraits::value_type, int>);
    static_assert(cuda::std::same_as<typename IterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::random_access_iterator<Iter>);
    static_assert(cuda::std::__is_cpp17_random_access_iterator<Iter>);
  }
  {
    // Member typedefs for random access iterator, LWG3798 rvalue reference.
    using Iter       = cuda::transform_iterator<random_access_iterator<int*>, IncrementRvalueRef>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::same_as<typename IterTraits::iterator_category, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<typename IterTraits::value_type, int>);
    static_assert(cuda::std::same_as<typename IterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::random_access_iterator<Iter>);
    static_assert(cuda::std::__is_cpp17_random_access_iterator<Iter>);
  }
  {
    // Member typedefs for random access iterator/not-lvalue-ref.
    using Iter       = cuda::transform_iterator<random_access_iterator<int*>, PlusOneMutable>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::same_as<typename IterTraits::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<typename IterTraits::value_type, int>);
    static_assert(cuda::std::same_as<typename IterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::random_access_iterator<Iter>);
    static_assert(cuda::std::__is_cpp17_random_access_iterator<Iter>);
  }
  {
    // Member typedefs for bidirectional iterator.
    using Iter       = cuda::transform_iterator<bidirectional_iterator<int*>, Increment>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::same_as<typename IterTraits::iterator_category, cuda::std::bidirectional_iterator_tag>);
    static_assert(cuda::std::same_as<typename IterTraits::value_type, int>);
    static_assert(cuda::std::same_as<typename IterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::bidirectional_iterator<Iter>);
    static_assert(cuda::std::__is_cpp17_bidirectional_iterator<Iter>);
  }
  {
    // Member typedefs for forward iterator.
    using Iter       = cuda::transform_iterator<forward_iterator<int*>, Increment>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::same_as<typename IterTraits::iterator_category, cuda::std::forward_iterator_tag>);
    static_assert(cuda::std::same_as<typename IterTraits::value_type, int>);
    static_assert(cuda::std::same_as<typename IterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::forward_iterator<Iter>);
    static_assert(cuda::std::__is_cpp17_forward_iterator<Iter>);
  }
  { // Nopthing to do here
    using Iter = cuda::transform_iterator<cpp17_input_iterator<int*>, Increment>;
    static_assert(!HasIterCategory<cpp17_input_iterator<int*>, Increment>);
    static_assert(cuda::std::__is_cpp17_input_iterator<Iter>);
  }

  {
    // Ensure we can work with other cuda iterators
    using Iter       = cuda::transform_iterator<cuda::counting_iterator<int>, TimesTwo>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::same_as<typename IterTraits::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<typename IterTraits::value_type, int>);
    static_assert(cuda::std::same_as<typename IterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::random_access_iterator<Iter>);
    static_assert(cuda::std::__is_cpp17_random_access_iterator<Iter>);
  }

  {
    // Ensure we can work with other cuda iterators
    using Iter       = cuda::std::reverse_iterator<cuda::transform_iterator<cuda::counting_iterator<int>, TimesTwo>>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::same_as<typename IterTraits::iterator_category, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<typename IterTraits::value_type, int>);
    static_assert(cuda::std::same_as<typename IterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::random_access_iterator<Iter>);
    static_assert(cuda::std::__is_cpp17_random_access_iterator<Iter>);
  }

  return true;
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
