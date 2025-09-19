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

__host__ __device__ void test()
{
  {
    using Iter = cuda::shuffle_iterator<char, fake_bijection<>>;
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::value_type, char>);
    static_assert(cuda::std::is_same_v<Iter::difference_type, cuda::std::make_signed_t<Iter::value_type>>);
    static_assert(cuda::std::is_signed_v<Iter::difference_type>);
    static_assert(cuda::std::same_as<Iter::difference_type, signed char>);
    static_assert(cuda::std::random_access_iterator<Iter>);
  }

  {
    using Iter = cuda::shuffle_iterator<short, fake_bijection<>>;
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::value_type, short>);
    static_assert(cuda::std::is_same_v<Iter::difference_type, cuda::std::make_signed_t<Iter::value_type>>);
    static_assert(cuda::std::is_signed_v<Iter::difference_type>);
    static_assert(cuda::std::same_as<Iter::difference_type, short>);
    static_assert(cuda::std::random_access_iterator<Iter>);
  }

  {
    using Iter = cuda::shuffle_iterator<size_t, fake_bijection<>>;
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::value_type, size_t>);
    static_assert(cuda::std::is_same_v<Iter::difference_type, cuda::std::make_signed_t<Iter::value_type>>);
    static_assert(cuda::std::is_signed_v<Iter::difference_type>);
    static_assert(cuda::std::same_as<Iter::difference_type, ptrdiff_t>);
    static_assert(cuda::std::random_access_iterator<Iter>);
  }
}

int main(int, char**)
{
  return 0;
}
