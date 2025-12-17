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
    using Iter = cuda::tabulate_output_iterator<basic_functor>;
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::value_type, void>);
    static_assert(cuda::std::same_as<Iter::difference_type, cuda::std::ptrdiff_t>);
#if !_CCCL_ARCH(ARM64) // There are some compiler issues on arm compilers
    static_assert(cuda::std::output_iterator<Iter, int>);
#endif // !_CCCL_ARCH(ARM64)
    static_assert(cuda::std::is_trivially_copyable_v<Iter>);
  }

  {
    using Iter = cuda::tabulate_output_iterator<basic_functor, signed char>;
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::value_type, void>);
    static_assert(cuda::std::same_as<Iter::difference_type, signed char>);
#if !_CCCL_ARCH(ARM64) // There are some compiler issues on arm compilers
    static_assert(cuda::std::output_iterator<Iter, int>);
#endif // !_CCCL_ARCH(ARM64)
    static_assert(cuda::std::is_trivially_copyable_v<Iter>);
  }
}

int main(int, char**)
{
  return 0;
}
