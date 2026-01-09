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
#include <cuda/std/type_traits>

#include "test_macros.h"
#include "types.h"

__host__ __device__ void test()
{
  {
    using Iter = cuda::strided_iterator<int*, int>;
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::value_type, int>);
    static_assert(cuda::std::is_signed_v<Iter::difference_type>);
    static_assert(cuda::std::same_as<Iter::difference_type, cuda::std::iter_difference_t<int*>>);
    static_assert(cuda::std::same_as<Iter::reference, cuda::std::iter_reference_t<int*>>);
    static_assert(cuda::std::random_access_iterator<Iter>);
    static_assert(cuda::std::is_trivially_copyable_v<Iter>);
  }

  {
    using Iter = cuda::strided_iterator<int*, Stride<2>>;
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::value_type, int>);
    static_assert(cuda::std::is_signed_v<Iter::difference_type>);
    static_assert(cuda::std::same_as<Iter::difference_type, cuda::std::iter_difference_t<int*>>);
    static_assert(cuda::std::same_as<Iter::reference, cuda::std::iter_reference_t<int*>>);
    static_assert(cuda::std::random_access_iterator<Iter>);
    static_assert(cuda::std::is_trivially_copyable_v<Iter>);
  }
}

int main(int, char**)
{
  return 0;
}
