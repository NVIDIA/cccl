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

#include "test_iterators.h"
#include "test_macros.h"

__host__ __device__ void test()
{
  {
    using baseIter             = random_access_iterator<int*>;
    using permutation_iterator = cuda::permutation_iterator<baseIter, baseIter>;
    using iterTraits           = cuda::std::iterator_traits<permutation_iterator>;

    static_assert(cuda::std::same_as<iterTraits::iterator_category, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<iterTraits::value_type, int>);
    static_assert(cuda::std::same_as<iterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::same_as<iterTraits::pointer, void>);
    static_assert(cuda::std::same_as<iterTraits::reference, int&>);
  }
  { // still random access
    using baseIter             = contiguous_iterator<int*>;
    using permutation_iterator = cuda::permutation_iterator<baseIter, baseIter>;
    using iterTraits           = cuda::std::iterator_traits<permutation_iterator>;

    static_assert(cuda::std::same_as<iterTraits::iterator_category, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<iterTraits::value_type, int>);
    static_assert(cuda::std::same_as<iterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::same_as<iterTraits::pointer, void>);
    static_assert(cuda::std::same_as<iterTraits::reference, int&>);
  }
}

int main(int, char**)
{
  return 0;
}
