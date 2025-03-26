//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// iterator_type, value_type, difference_type, iterator_concept, iterator_category

#include <cuda/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

// No value_type.
struct InputOrOutputArchetype
{
  using difference_type = int;

  __host__ __device__ int operator*();
  __host__ __device__ void operator++(int);
  __host__ __device__ InputOrOutputArchetype& operator++();
};

template <class T>
_CCCL_CONCEPT HasValueType = _CCCL_REQUIRES_EXPR((T))(typename(typename T::value_type));

template <class T>
_CCCL_CONCEPT HasIteratorConcept = _CCCL_REQUIRES_EXPR((T))(typename(typename T::iterator_concept));

template <class T>
_CCCL_CONCEPT HasIteratorCategory = _CCCL_REQUIRES_EXPR((T))(typename(typename T::iterator_category));

__host__ __device__ void test()
{
  {
    using Iter = cuda::std::counted_iterator<InputOrOutputArchetype>;
    static_assert(cuda::std::same_as<Iter::iterator_type, InputOrOutputArchetype>);
    static_assert(!HasValueType<Iter>);
    static_assert(cuda::std::same_as<Iter::difference_type, int>);
    static_assert(!HasIteratorConcept<Iter>);
    static_assert(!HasIteratorCategory<Iter>);
  }
  {
    using Iter = cuda::std::counted_iterator<cpp20_input_iterator<int*>>;
    static_assert(cuda::std::same_as<Iter::iterator_type, cpp20_input_iterator<int*>>);
    static_assert(cuda::std::same_as<Iter::value_type, int>);
    static_assert(cuda::std::same_as<Iter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::input_iterator_tag>);
    static_assert(!HasIteratorCategory<Iter>);
  }
  {
    using Iter = cuda::std::counted_iterator<random_access_iterator<int*>>;
    static_assert(cuda::std::same_as<Iter::iterator_type, random_access_iterator<int*>>);
    static_assert(cuda::std::same_as<Iter::value_type, int>);
    static_assert(cuda::std::same_as<Iter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(!HasIteratorConcept<Iter>);
    static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::random_access_iterator_tag>);
  }
}

int main(int, char**)
{
  return 0;
}
