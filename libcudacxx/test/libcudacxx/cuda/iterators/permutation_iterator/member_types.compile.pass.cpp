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
#include <cuda/std/type_traits>

#include "test_iterators.h"
#include "test_macros.h"

__host__ __device__ void test()
{
  {
    using Iter = cuda::permutation_iterator<int*>;
    static_assert(cuda::std::same_as<Iter::iterator_type, int*>);
    static_assert(cuda::std::same_as<Iter::value_type, int>);
    static_assert(cuda::std::same_as<Iter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::is_trivially_copyable_v<Iter>);
  }
  {
    using Iter = cuda::permutation_iterator<int*, random_access_iterator<int*>>;
    static_assert(cuda::std::same_as<Iter::iterator_type, int*>);
    static_assert(cuda::std::same_as<Iter::value_type, int>);
    static_assert(cuda::std::same_as<Iter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::is_trivially_copyable_v<Iter>);
  }

  { // We take the value_type and iterator_type but difference type comes from the index iterator
    using Iter = cuda::permutation_iterator<cuda::counting_iterator<short>, int*>;
    static_assert(cuda::std::same_as<Iter::iterator_type, cuda::counting_iterator<short>>);
    static_assert(cuda::std::same_as<Iter::value_type, short>);
    static_assert(cuda::std::same_as<Iter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::is_trivially_copyable_v<Iter>);
  }

  { // We take the value_type and iterator_type but difference type comes from the index iterator
    using Iter = cuda::permutation_iterator<int*, cuda::counting_iterator<short>>;
    static_assert(cuda::std::same_as<Iter::iterator_type, int*>);
    static_assert(cuda::std::same_as<Iter::value_type, int>);
    static_assert(cuda::std::same_as<Iter::difference_type, int>);
    static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::is_trivially_copyable_v<Iter>);
  }
}

int main(int, char**)
{
  return 0;
}
