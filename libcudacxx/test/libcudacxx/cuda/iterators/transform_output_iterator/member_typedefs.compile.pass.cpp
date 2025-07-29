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

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

__host__ __device__ void test()
{
  {
    using Iter = cuda::transform_output_iterator<int*, PlusOne>;
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::output_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::output_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::pointer, void>);
    static_assert(cuda::std::same_as<Iter::reference, void>);
    static_assert(cuda::std::same_as<Iter::value_type, void>);
    static_assert(cuda::std::same_as<Iter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::output_iterator<Iter, int>);
  }

  {
    using Iter = cuda::transform_output_iterator<random_access_iterator<int*>, PlusOne>;
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::output_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::output_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::pointer, void>);
    static_assert(cuda::std::same_as<Iter::reference, void>);
    static_assert(cuda::std::same_as<Iter::value_type, void>);
    static_assert(cuda::std::same_as<Iter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::output_iterator<Iter, int>);
  }
}

int main(int, char**)
{
  return 0;
}
