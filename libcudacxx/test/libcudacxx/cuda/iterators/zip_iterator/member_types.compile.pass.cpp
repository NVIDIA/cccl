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

template <class T>
_CCCL_CONCEPT HasIterCategory = _CCCL_REQUIRES_EXPR((T))(typename(typename T::iterator_category));

template <class T>
struct DiffTypeIter
{
  using iterator_category = cuda::std::input_iterator_tag;
  using value_type        = int;
  using difference_type   = T;

  __host__ __device__ int operator*() const;
  __host__ __device__ DiffTypeIter& operator++();
  __host__ __device__ void operator++(int);
#if TEST_STD_VER >= 2020
  __host__ __device__ friend constexpr bool operator==(DiffTypeIter, DiffTypeIter) = default;
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
  __host__ __device__ friend constexpr bool operator==(const DiffTypeIter&, const DiffTypeIter&)
  {
    return true;
  }
  __host__ __device__ friend constexpr bool operator!=(const DiffTypeIter&, const DiffTypeIter&)
  {
    return false;
  }
#endif // TEST_STD_VER <=2017
};

struct Foo
{};

__host__ __device__ void test()
{
  { // Single iterator should have tuple value type
    using Iter = cuda::zip_iterator<int*>;
    static_assert(cuda::std::is_same_v<Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<Iter::value_type, cuda::std::tuple<int>>);
    static_assert(HasIterCategory<Iter>);
    static_assert(cuda::std::random_access_iterator<Iter>);
  }

  { // Two iterator should have pair value type
    using Iter = cuda::zip_iterator<int*, Foo*>;
    static_assert(cuda::std::is_same_v<Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<Iter::value_type, cuda::std::pair<int, Foo>>);
    static_assert(HasIterCategory<Iter>);
    static_assert(cuda::std::random_access_iterator<Iter>);
  }

  { // !=2 views should have tuple value_type
    using Iter = cuda::zip_iterator<int*, Foo*, int*>;
    static_assert(cuda::std::is_same_v<Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<Iter::value_type, cuda::std::tuple<int, Foo, int>>);
    static_assert(HasIterCategory<Iter>);
    static_assert(cuda::std::random_access_iterator<Iter>);
  }

  { // If one iterator is not random access then the whole zip_iterator is not random access
    using Iter = cuda::zip_iterator<int*, Foo*, bidirectional_iterator<int*>>;
    static_assert(cuda::std::is_same_v<Iter::iterator_concept, cuda::std::bidirectional_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<Iter::value_type, cuda::std::tuple<int, Foo, int>>);
    static_assert(HasIterCategory<Iter>);
    static_assert(cuda::std::bidirectional_iterator<Iter>);
  }

  { // If one iterator is not bidirectional_iterator then the whole zip_iterator is not bidirectional_iterator
    using Iter = cuda::zip_iterator<forward_iterator<int*>, Foo*, bidirectional_iterator<int*>>;
    static_assert(cuda::std::is_same_v<Iter::iterator_concept, cuda::std::forward_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<Iter::value_type, cuda::std::tuple<int, Foo, int>>);
    static_assert(HasIterCategory<Iter>);
    static_assert(cuda::std::forward_iterator<Iter>);
  }

  { // If one iterator is not forward_iterator then the whole zip_iterator is not forward_iterator
    using Iter = cuda::zip_iterator<forward_iterator<int*>, cpp20_input_iterator<Foo*>, bidirectional_iterator<int*>>;
    static_assert(cuda::std::is_same_v<Iter::iterator_concept, cuda::std::input_iterator_tag>);
    static_assert(!HasIterCategory<Iter>);
    static_assert(cuda::std::is_same_v<Iter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<Iter::value_type, cuda::std::tuple<int, Foo, int>>);
    static_assert(cuda::std::input_iterator<Iter>);
  }

  { // nested iterator has the right value type
    using Iter = cuda::zip_iterator<int*, cuda::zip_iterator<Foo*, int*>>;
    static_assert(cuda::std::is_same_v<Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<Iter::value_type, cuda::std::pair<int, cuda::std::pair<Foo, int>>>);
    static_assert(HasIterCategory<Iter>);
    static_assert(cuda::std::random_access_iterator<Iter>);
  }

  { // Takes the difference type from the base iterator
    using Iter = cuda::zip_iterator<DiffTypeIter<intptr_t>>;
    static_assert(cuda::std::is_same_v<Iter::difference_type, intptr_t>);
  }

  { // Difference type is the common type of the difference types
    using Iter = cuda::zip_iterator<DiffTypeIter<intptr_t>, DiffTypeIter<cuda::std::ptrdiff_t>>;
    static_assert(
      cuda::std::is_same_v<Iter::difference_type, cuda::std::common_type_t<intptr_t, cuda::std::ptrdiff_t>>);
  }
}

int main(int, char**)
{
  return 0;
}
