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

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

__host__ __device__ void test()
{
  {
    using Iter = cuda::transform_output_iterator<PlusOne, int*>;
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::output_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::pointer, void>);
    static_assert(cuda::std::same_as<Iter::reference, void>);
    static_assert(cuda::std::same_as<Iter::value_type, void>);
    static_assert(cuda::std::same_as<Iter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::output_iterator<Iter, int>);
    static_assert(cuda::std::is_trivially_copyable_v<Iter>);
  }

  {
    using Iter = cuda::transform_output_iterator<PlusOne, random_access_iterator<int*>>;
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::output_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::pointer, void>);
    static_assert(cuda::std::same_as<Iter::reference, void>);
    static_assert(cuda::std::same_as<Iter::value_type, void>);
    static_assert(cuda::std::same_as<Iter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::output_iterator<Iter, int>);
    static_assert(cuda::std::is_trivially_copyable_v<Iter>);
  }

  {
    using Iter = cuda::transform_output_iterator<PlusOne, bidirectional_iterator<int*>>;
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::bidirectional_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::output_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::pointer, void>);
    static_assert(cuda::std::same_as<Iter::reference, void>);
    static_assert(cuda::std::same_as<Iter::value_type, void>);
    static_assert(cuda::std::same_as<Iter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::output_iterator<Iter, int>);
    static_assert(cuda::std::is_trivially_copyable_v<Iter>);
  }

  {
    using Iter = cuda::transform_output_iterator<PlusOne, forward_iterator<int*>>;
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::forward_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::output_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::pointer, void>);
    static_assert(cuda::std::same_as<Iter::reference, void>);
    static_assert(cuda::std::same_as<Iter::value_type, void>);
    static_assert(cuda::std::same_as<Iter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::output_iterator<Iter, int>);
    static_assert(cuda::std::is_trivially_copyable_v<Iter>);
  }
}

int main(int, char**)
{
  return 0;
}
