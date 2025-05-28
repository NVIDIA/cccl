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

#include <cuda/iterator>

#include "test_iterators.h"
#include "test_macros.h"

template <class T>
_CCCL_CONCEPT HasValueType = _CCCL_REQUIRES_EXPR((T))(typename(typename T::value_type));

template <class T>
_CCCL_CONCEPT HasIteratorConcept = _CCCL_REQUIRES_EXPR((T))(typename(typename T::iterator_concept));

template <class T>
_CCCL_CONCEPT HasIteratorCategory = _CCCL_REQUIRES_EXPR((T))(typename(typename T::iterator_category));

__host__ __device__ void test()
{
  {
    using Iter = cuda::offset_iterator<int*>;
    static_assert(cuda::std::same_as<Iter::iterator_type, int*>);
    static_assert(cuda::std::same_as<Iter::value_type, int>);
    static_assert(cuda::std::same_as<Iter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(HasIteratorConcept<Iter>);
    static_assert(HasIteratorCategory<Iter>);
  }
  {
    using Iter = cuda::offset_iterator<random_access_iterator<int*>>;
    static_assert(cuda::std::same_as<Iter::iterator_type, random_access_iterator<int*>>);
    static_assert(cuda::std::same_as<Iter::value_type, int>);
    static_assert(cuda::std::same_as<Iter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(HasIteratorConcept<Iter>);
    static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::random_access_iterator_tag>);
  }
}

int main(int, char**)
{
  return 0;
}
