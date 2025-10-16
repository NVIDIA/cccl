//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// Iterator traits and member typedefs in zip_view::<iterator>.

#include <cuda/iterator>
#include <cuda/std/tuple>

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

#if !TEST_COMPILER(NVRTC)
#  include <iterator>
#endif // !TEST_COMPILER(NVRTC)

template <class T>
_CCCL_CONCEPT HasIterCategory = _CCCL_REQUIRES_EXPR((T))(typename(typename T::iterator_category));

struct Foo
{};

template <template <class...> class Traits>
__host__ __device__ void test()
{
  { // Single iterator should have tuple value type
    using Iter       = cuda::zip_iterator<int*>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::is_same_v<typename IterTraits::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::is_same_v<typename IterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<typename IterTraits::value_type, cuda::std::tuple<int>>);
    static_assert(cuda::std::random_access_iterator<Iter>);
    static_assert(cuda::std::__has_random_access_traversal<Iter>);
  }

  { // Two iterator should have pair value type
    using Iter       = cuda::zip_iterator<int*, Foo*>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::is_same_v<typename IterTraits::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::is_same_v<typename IterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<typename IterTraits::value_type, cuda::std::tuple<int, Foo>>);
    static_assert(cuda::std::random_access_iterator<Iter>);
    static_assert(cuda::std::__has_random_access_traversal<Iter>);
  }

  { // !=2 views should have tuple value_type
    using Iter       = cuda::zip_iterator<int*, Foo*, int*>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::is_same_v<typename IterTraits::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::is_same_v<typename IterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<typename IterTraits::value_type, cuda::std::tuple<int, Foo, int>>);
    static_assert(cuda::std::random_access_iterator<Iter>);
    static_assert(cuda::std::__has_random_access_traversal<Iter>);
  }

  { // If one iterator is not random access then the whole zip_iterator is not random access
    using Iter       = cuda::zip_iterator<int*, Foo*, bidirectional_iterator<int*>>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::is_same_v<typename IterTraits::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::is_same_v<typename IterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<typename IterTraits::value_type, cuda::std::tuple<int, Foo, int>>);
    static_assert(cuda::std::bidirectional_iterator<Iter>);
    static_assert(cuda::std::__has_bidirectional_traversal<Iter>);
  }

  { // If one iterator is not bidirectional_iterator then the whole zip_iterator is not bidirectional_iterator
    using Iter       = cuda::zip_iterator<forward_iterator<int*>, Foo*, bidirectional_iterator<int*>>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::is_same_v<typename IterTraits::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::is_same_v<typename IterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<typename IterTraits::value_type, cuda::std::tuple<int, Foo, int>>);
    static_assert(cuda::std::forward_iterator<Iter>);
    static_assert(cuda::std::__has_forward_traversal<Iter>);
  }

  { // Nothing here
    using Iter = cuda::zip_iterator<forward_iterator<int*>, cpp20_input_iterator<Foo*>, bidirectional_iterator<int*>>;
    static_assert(!HasIterCategory<Iter>);
    static_assert(cuda::std::__has_input_traversal<Iter>);
  }

  { // nested iterator has the right value type
    using Iter       = cuda::zip_iterator<int*, cuda::zip_iterator<Foo*, int*>>;
    using IterTraits = Traits<Iter>;
    static_assert(cuda::std::is_same_v<typename IterTraits::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::is_same_v<typename IterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(
      cuda::std::is_same_v<typename IterTraits::value_type, cuda::std::tuple<int, cuda::std::tuple<Foo, int>>>);
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
